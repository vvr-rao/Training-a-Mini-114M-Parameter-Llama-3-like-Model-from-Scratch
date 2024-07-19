import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Tuple

from ModelParams import *


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def create_causal_mask(seq_len: int, device: str):
  mask = torch.full(
    (seq_len, seq_len),
    float("-inf")
  )
  mask = torch.triu(mask, diagonal=1)
  return(mask.to(device))


def precompute_rotary_embeddings(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
  #default theta set to 10,000. recomended value

  #STEP 1: Compute the theta_i values
  #formula for theta_i(freqs) is theta ^ (-(2(i - 1)/dim)) where i = 1, 2, ... dim/2
  #torch.arrange gived number from 0 to end integer

  freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
  #freqs shape will be size head_dim/2

  #print(f'freqs: {freqs.shape}\n{freqs}\n')

  #STEP 2: Compute the m_i * theta_i matrix
  #just an array of numbers from 0 to 2xseq_length
  # why 2xseq_length? - so we can project into the generated tokens
  t = torch.arange(seq_len, dtype=torch.float32)
  #print(f't: {t.shape}\n{t}\n')

  freqs = torch.outer(t, freqs)   #outer product of tensors
  #print(f'freqs: {freqs.shape}\n{freqs}\n')
  #this shape will be a matrix of 2xseq_length, head_dim

  # STEP 3: return the cosing and sine values for the m_i * theta_i computed earlier
  # torch.ones_like(freqs)  #returns tensor of ones with same shape as freqs
  # torch.polar(abs, angle) - returns a complex tensor - abs⋅cos(angle)+abs⋅sin(angle)⋅j

  freqs_complex = torch.polar(torch.ones_like(freqs), freqs)[:seq_len]
  #print(f'freqs_complex: {freqs_complex.shape}\n{freqs_complex}')

  return(freqs_complex.to(device))


def apply_rotary_embeddings(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, ) -> Tuple[torch.Tensor, torch.Tensor]:
  #STEP 1: group pairs of the q and k tensors and move them into complex numbers e.f. [x1, x2, x3, x4] => [[x1 + ix2], [x3 + ix4]]
  # we only apply the ROPE Embeddings to the q and k matrices, not the v matrix
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

  #STEP 2: reshape the precomputed ROPE Embeddings into a form we can multiply with
  ndim = xq_.ndim
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
  freqs_cis = freqs_cis.view(*shape)

  #STEP 3: Multiply the reshpaed q and k matrices and reshape back into the original shape
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

  return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, device):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        attn_shape = (self.n_heads + 2 * args.n_kv_heads) * self.head_dim

        # Attention Weight Matrix
        self.qkv_attn = nn.Linear(args.dim, attn_shape, bias=False)
        #Output Weight Matrix
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)


        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            requires_grad = False
        ).to(device)
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            requires_grad = False
        ).to(device)

    def forward(self,  x: torch.Tensor,  freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], start_pos: int = None,  ):
        num_batches, seqlen, _ = x.shape
        q_per_kv = self.n_heads // self.n_kv_heads

        
        # e.g. for dim = 768, num_q_heads = 12, num_kv_heads = 4
        # this would create a matix with dimensions B, T, (768 + 256 + 256)
        qkv = self.qkv_attn(x)

        # create a view with B, T, 4, 5, 64, (768/12)
        qkv = qkv.view(num_batches, seqlen, self.n_kv_heads, q_per_kv + 2, self.head_dim)

        #now split
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        q = q.reshape(num_batches, seqlen, -1, self.head_dim)  # (B, T, nh_q, hs)
        k = k.reshape(num_batches, seqlen, -1, self.head_dim)  
        v = v.reshape(num_batches, seqlen, -1, self.head_dim)  

        q, k = apply_rotary_embeddings(q, k, freqs_cis=freqs_cis)

        if start_pos is not None:
            #KV CACHE FOR INFERENCE
            #this is from the LLAMA3 code
            # make sure our cache is on the right device
            self.cache_k = self.cache_k.to(q)
            self.cache_v = self.cache_v.to(q)

            # set the values in our cache according to the current input
            self.cache_k[:num_batches, start_pos : start_pos + seqlen] = k
            self.cache_v[:num_batches, start_pos : start_pos + seqlen] = v

            # grab our key and value matrixes which have a longer sequence length than our queries
            keys = self.cache_k[:num_batches, : start_pos + seqlen]
            values = self.cache_v[:num_batches, : start_pos + seqlen]
        else:
            # TRAINING
            keys, values = k, v

        scale = 1/q.shape[-1]**0.5
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        #automatically uses flash attention and applies softmax scale
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=mask is None) 
        #o = o.transpose(1,2).contiguous().view(num_batches, seqlen, self.dim)
        o = o.transpose(1,2).reshape(num_batches, seqlen, self.dim)
        
        


        return self.wo(o)
