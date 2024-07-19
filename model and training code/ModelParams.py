import torch
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelArgs:
    dim: int = 768 #256 #128
    n_layers: int = 12 # Llama 3 8b uses 32
    n_heads: int = 12 # Llama 3 8b uses 32
    n_kv_heads: Optional[int] = 4 # Llama 3 8b's uses 8
    vocab_size: int = 50304 #512 # Llama 3 uses a more complicated tokenizer of length 128256
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2. Llama 3 8b's uses 1024
    ffn_dim_multiplier: Optional[float] = None # Llama 3 8b's uses 1.3, which changes the ending hidden_dim slightly
    norm_eps: float = 1e-5
    rope_theta: float = 10000 # Llama 3 8b uses 500000
    max_batch_size: int = 16
    grad_accum_steps: int = 4
    max_seq_len: int = 1024 #256 # Llama 3 8b trained with 8192 but their maximum kv cache chunk size during inference is 2048
    dropout_rate: float = 0.1 
    bfloat_supported: bool = True #does the machine support bfloat16 for autocast
    flash_attn_supported: bool = False #does the mahine support Flahs Attention?
    
