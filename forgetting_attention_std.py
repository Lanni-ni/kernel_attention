

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


def forgetting_attention_std(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_fgate: torch.Tensor,
    *,
    head_first: bool = False,
    seq_start: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:

    
    if not head_first:
        q = rearrange(q, "b t h d -> b h t d")
        k = rearrange(k, "b t h d -> b h t d")
        v = rearrange(v, "b t h d -> b h t d")
        log_fgate = rearrange(log_fgate, "b t h -> b h t")
    
    B, H, T_q, D = q.shape
    T_k = k.shape[2]
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    
 
    log_fgate_masked = log_fgate.float()
    if seq_start is not None:
        log_fgate_masked = log_fgate_masked.clone()
        mask_idx = torch.arange(T_k, device=q.device)[None, None, :] < seq_start[:, None, None]
        log_fgate_masked[mask_idx] = 0.0
    
    log_lambda = torch.cumsum(log_fgate_masked, dim=-1)
    decay_bias = log_lambda[:, :, :T_q, None] - log_lambda[:, :, None, :]
    scores = scores + decay_bias
    
    # Causal mask
    P_SEQ = T_k - T_q
    causal_mask = torch.triu(torch.ones((T_q, T_k), dtype=torch.bool, device=q.device), diagonal=P_SEQ + 1)
    scores = scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))
    
    # seq_start mask
    if seq_start is not None:
        seq_mask = torch.arange(T_k, device=q.device)[None, None, None, :] < seq_start[None, :, None, None]
        scores = scores.masked_fill(seq_mask, float('-inf'))
    
    # Softmax
    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, 0.0)
    
 
    out = torch.matmul(attn.to(v.dtype), v)
    
    if not head_first:
        out = rearrange(out, "b h t d -> b t h d")
    
    return out
