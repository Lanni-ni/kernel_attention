"""
Stick-breaking Attention - 官方Triton实现
"""

from stickbreaking_attention.sb_attn import sb_attn
import math
import torch
from einops import rearrange
from typing import Optional


def stickbreaking_attention_std(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    head_first: bool = False,
    seq_start: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    normalize: bool = True,
    attend_current: bool = False,
) -> torch.Tensor:
    """Stick-breaking attention using official Triton implementation"""
    
    if not head_first:
        q = rearrange(q, "b t h d -> b h t d")
        k = rearrange(k, "b t h d -> b h t d")
        v = rearrange(v, "b t h d -> b h t d")
    
    B, H, T_q, D = q.shape
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    
    # 官方Triton实现
    # 返回 (output, remainder)
    out, rem = sb_attn(
        q, k, v, 
        inv_temp=sm_scale,
        attend_current=attend_current
    )
    
    if not head_first:
        out = rearrange(out, "b h t d -> b t h d")
    
    return out
