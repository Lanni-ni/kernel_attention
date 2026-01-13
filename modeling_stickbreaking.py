# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from fla.modules import FusedCrossEntropyLoss, RMSNorm, RotaryEmbedding
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationConfig

from .configuration_stickbreaking import StickbreakingConfig


class StickbreakingAttention(nn.Module):
    """
    Stick-breaking attention mechanism (ICLR 2025)
    """
    
    def __init__(self, config: StickbreakingConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Q, K, V projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Optional: RoPE
        if config.use_rope:
            self.rotary = RotaryEmbedding(
                dim=self.head_dim,
                base=config.rope_base
            )
        
        # Optional: QK norm
        if config.qk_norm:
            if config.qk_norm_share_param_across_head:
                self.q_norm = RMSNorm(hidden_size=self.head_dim, eps=config.norm_eps)
                self.k_norm = RMSNorm(hidden_size=self.head_dim, eps=config.norm_eps)
            else:
                self.q_norm = RMSNorm(hidden_size=self.hidden_size, eps=config.norm_eps)
                self.k_norm = RMSNorm(hidden_size=self.num_kv_heads * self.head_dim, eps=config.norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.size()
        
        # QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Optional: RoPE
        if self.config.use_rope:
            q, k = self.rotary(q, k)
        
        # Optional: QK norm
        if self.config.qk_norm:
            if self.config.qk_norm_share_param_across_head:
                q = self.q_norm(q)
                k = self.k_norm(k)
            else:
                q = self.q_norm(q.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
                k = self.k_norm(k.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
                q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat K, V if using GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Stick-breaking attention
        from forgetting_transformer.ops.stickbreaking_attention_std import stickbreaking_attention_std
        
        o = stickbreaking_attention_std(
            q, k, v,
            head_first=True,
            sm_scale=self.scale,
            normalize=self.config.normalize_attention,
            attend_current=self.config.attend_current,
        )
        
        # Output projection
        o = o.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        o = self.o_proj(o)
        
        return o, None


class StickbreakingMLP(nn.Module):
    def __init__(self, config: StickbreakingConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size or config.hidden_ratio * config.hidden_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class StickbreakingBlock(nn.Module):
    def __init__(self, config: StickbreakingConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.attn = StickbreakingAttention(config, layer_idx)
        
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.mlp = StickbreakingMLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        # Attention with residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, present_key_value = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class StickbreakingPreTrainedModel(PreTrainedModel):
    config_class = StickbreakingConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["StickbreakingBlock"]
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class StickbreakingModel(StickbreakingPreTrainedModel):
    def __init__(self, config: StickbreakingConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            StickbreakingBlock(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        hidden_states = self.embeddings(input_ids)
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, _ = torch.utils.checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    None,
                    use_cache,
                )
            else:
                hidden_states, _ = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=None,
                    use_cache=use_cache,
                )
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class StickbreakingForCausalLM(StickbreakingPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = StickbreakingModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embeddings
    
    def set_input_embeddings(self, value):
        self.model.embeddings = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through model
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss
        loss = None
        if labels is not None:
            if self.config.fuse_cross_entropy:
                loss_fct = FusedCrossEntropyLoss(inplace_backward=True, reduction='none')
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='none')
            
            logits = logits.to(torch.float32)
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss = loss.view(*labels.size())
        
        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )