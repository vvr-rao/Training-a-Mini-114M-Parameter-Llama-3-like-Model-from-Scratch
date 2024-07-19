# simple launch:
# python train.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train.py


#when calling DDP through torchrun, the code will be called on each GPU.
#Important to set the seed to ensure the models are initialized the same across GPUs
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# import tokenizer
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
eot = tokenizer._special_tokens['<|endoftext|>'] # end of text token

#import other libraries
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import time

import os
import json

from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
#from torch.utils.data.distributed import DistributedSampler not needed since we will not use Dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os

#from einops import rearrange, einsum
from Model import *
from S3Helper import *

import numpy as np


# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

#DataLoader code
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        #if master_process:
        print(f"found {len(shards)} shards for split {split} on device {device}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


B = params.max_batch_size #16
T = params.max_seq_len #1024
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")


#instantiate model
model = LLM(params, tokenizer, device)
model = model.to(device)

#torch.compile
#model = torch.compile(model)  #use torch.compile to speed things up

#wrap the model in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

if master_process:
    s = sum(p.numel() for p in model.parameters())
    print(f"{s} parameter model trained on 10 Billion tokens")


#the Evaluator

@torch.no_grad()
def estimate_val_loss(model, batch_size, eval_iters = 10): # to estimate loss during the training loop
    out = {}
    model.eval() # sets model to eval mode
    #for split in ['train', 'val']:
    for split in ['val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            #X, Y = get_batch(split, batch_size)
            #logits, loss = model(X, targets=Y)
            #losses[k] = loss.item()
            #dl = get_dataloader(split, batch_size)
            #for s, t in dl:
            s, t = val_loader.next_batch()
            s, t = s.to(device), t.to(device)
            # if the GPU supports BFLOAT, autocast to BF16
            if (params.bfloat_supported):
              with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(s, targets=t)
                losses[k] = loss.item()
            else:
              logits, loss = model(s, targets=t)
              losses[k] = loss.item()

        out[split] = losses.mean()
    model.train() # just resets to training mode
    return out

#***************************************************
# how long we want to train for
#***************************************************
max_iters = 15000 #2000
# how often we want to check & see how our loss is doing
eval_interval = 50

# Warmup setup
warmup_iters = 500  # Number of warmup iterations
warmup_factor = 1e-3  # Warmup factor (initial learning rate is multiplied by this factor)

# create a PyTorch optimizer
lr_init = 6e-4
lr_final = lr_init * 0.1 # Minimum learning rate
weight_decay = 0.02
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)





def lr_lambda(current_iter):
    if current_iter < warmup_iters:
        # Warmup phase
        return warmup_factor + (1 - warmup_factor) * current_iter / warmup_iters
    else:
        # Cosine decay phase with minimum learning rate
        decay_iters = max_iters - warmup_iters
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - warmup_iters) / decay_iters))
        return max(cosine_decay, lr_final / lr_init)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


#TRAINING LOOP!!!

start_time = time.time()

# Enable anomaly detection. uncomment these lines if you need to do extensive debugging
#torch.autograd.set_detect_anomaly(True)

for iter in range(max_iters):
    loss_accum = 0.0 #this is just for display purposes
    model.train()
    for microstep in range(params.grad_accum_steps):
      #dl = get_dataloader('train', params.max_batch_size)
      #for s, t in dl:
      s, t = train_loader.next_batch()
      s, t = s.to(device), t.to(device)
      #This might not be the official way to do this. If thing break in the future, look here
      if ddp:
        #model.require_backward_grad_sync = (microstep == params.grad_accum_steps - 1)
        if (microstep == params.grad_accum_steps - 1):
            #print("synching")
            model.require_backward_grad_sync = True
      #if bfloat16 is supported, use it
      if (params.bfloat_supported):
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
          logits, loss = model(s, targets=t)
      else:
        logits, loss = model(s, targets=t)
      loss = loss / params.grad_accum_steps
      loss.backward()
      loss_accum += loss.detach()
      norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #clip gradients
      #optimizer.step() # update weights
      #optimizer.zero_grad() #REMEMBER TO ZERO THE GRADIENTS!!!
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    optimizer.step() # update weights
    optimizer.zero_grad() #REMEMBER TO ZERO THE GRADIENTS!!!
    # Update the learning rate
    scheduler.step()

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if master_process:
          #only estimate Val loss for the Master process
          losses = estimate_val_loss(model, params.max_batch_size)
          current_lr = optimizer.param_groups[0]['lr']
          print(f"step {iter:04d}: lr {current_lr:.6f}, train loss {loss_accum:.4f}, val loss {losses['val']:.4f}, time elapsed: {elapsed_time:.2f} seconds")


# Disable anomaly detection after the training loop
#torch.autograd.set_detect_anomaly(False)


#Save the model
if master_process:
    print("SAVING MODEL")
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    input_str = "Hello, I am Mr. LLM. I "
    output = raw_model.generate(
        input_str,
        max_gen_len = 50 + len(input_str),
        temperature = 0.6, 
        top_p = 0.9, 
        top_k = 32, 
    )
    print("----------------")
    print(output)

    input_str = "These are the symptoms of HIV. "
    output = raw_model.generate(
        input_str,
        max_gen_len = 50 + len(input_str),
        temperature = 0.6, 
        top_p = 0.9, 
        top_k = 32, 
    )
    print("----------------")
    print(output)

    input_str = "Here is a story about a pirate and a ninja. "
    output = raw_model.generate(
        input_str,
        max_gen_len = 200 + len(input_str),
        temperature = 0.6, 
        top_p = 0.9, 
        top_k = 32, 
    )
    print("----------------")
    print(output)

    os.makedirs("working", exist_ok=True)
    os.makedirs("download", exist_ok=True)
    torch.save(raw_model.state_dict(), './working/best_model_state-114m.bin')
    #upload to S3
    bucket_name = "trained-model-venkat"
    upload_file_to_s3("./working/best_model_state-114m.bin", bucket_name,"working/best_model_state-114m.bin")


    download_file_from_s3(bucket_name,"working/best_model_state-114m.bin", "./download/best_model_state.bin")
    loaded_model = LLM(params, tokenizer, device).to(device)
    loaded_model.load_state_dict(torch.load("./download/best_model_state.bin"))
    loaded_model.eval()

    input_str = "I am a Large Language Model. Therefore,"
    output = loaded_model.generate(
        input_str,
        max_gen_len = 100 - len(input_str), 
        temperature = 0.6, 
        top_p = 0.9, 
        top_k = 32, 
    )
    print("----------------")
    print(output)

    input_str = "Write a poem about a Walrus meeting a Carpenter at a beach. "

    output = loaded_model.generate(
        input_str,
        max_gen_len = 200 - len(input_str), 
        temperature = 0.6, 
        top_p = 0.9, 
        top_k = 32, 
    )
    print("----------------")
    print(output)

    input_str = "The pirate and the ninja were having coffee at the bar. They "
    output = loaded_model.generate(
        input_str,
        max_gen_len = 200 - len(input_str), 
        temperature = 0.6, 
        top_p = 0.9, 
        top_k = 32, 
    )
    print("----------------")
    print(output)
    
#exit gracefully
if ddp:
    destroy_process_group()


