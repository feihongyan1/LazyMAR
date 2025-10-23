"""
Basic building blocks for MAR (Masked Autoregressive) model.

This module contains the core components for both encoder and decoder:
- LayerScale: Learnable layer scaling
- EncoderAttention/EncoderBlock: Encoder components with incremental KV caching
- DecoderAttention/DecoderBlock: Decoder components with pruning and caching mechanisms
"""

import logging
import copy
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
try:
    from timm.layers import Mlp, DropPath, use_fused_attn
except ImportError:
    # For older versions of timm (< 0.9.0)
    from timm.models.layers import Mlp, DropPath
    def use_fused_attn() -> bool:
        return False


_logger = logging.getLogger(__name__)

# Retention ratio schedule for decoder pruning strategy
RETAIN_RATIO_SCHEDULE = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.6, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4,
    0.15, 0.15, 0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12,
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.05
]


class LayerScale(nn.Module):
    """
    Layer scaling module with learnable per-channel scaling factors.
    
    Args:
        dim: Feature dimension
        init_values: Initial scaling value
        inplace: Whether to perform inplace operations
    """
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class EncoderAttention(nn.Module):
    """
    Multi-head self-attention for encoder with incremental KV caching support.
    
    Supports lazy evaluation by accumulating key-value pairs across iterations.
    
    Args:
        dim: Input feature dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        qk_norm: Whether to apply normalization to Q and K
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
        norm_layer: Normalization layer type
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cur_kv_len = 0

    def forward(self, x: torch.Tensor, current: dict, cache_dic: dict) -> torch.Tensor:
        """
        Forward pass with incremental KV caching.
        
        Args:
            x: Input tensor of shape (B, N, C)
            current: Dictionary containing current state information
            cache_dic: Dictionary containing cached KV pairs
            
        Returns:
            Output tensor of shape (B, N, C)
        """
        self.cur_kv_len = cache_dic['enco_cache'][current['enco_layer_idx']]['cur_kv_len']
        
        if current['lazy_mar'] and not current['is_force_fresh']:
            # Incremental update mode
            B, N, C = x.shape
            app_kv_len = N - 64  # Append length (excluding buffer tokens)

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            
            # Move cached KV to same device as current KV
            cache_dic['enco_cache'][current['enco_layer_idx']]['k'].to(k)
            cache_dic['enco_cache'][current['enco_layer_idx']]['v'].to(k)

            # Append new KV pairs to cache
            cache_dic['enco_cache'][current['enco_layer_idx']]['k'][:, :, 64+self.cur_kv_len:64+self.cur_kv_len+app_kv_len, :] = k[:, :, 64:, :]
            cache_dic['enco_cache'][current['enco_layer_idx']]['k'][:, :, 0:64, :] = k[:, :, 0:64, :]

            cache_dic['enco_cache'][current['enco_layer_idx']]['v'][:, :, 64+self.cur_kv_len:64+self.cur_kv_len+app_kv_len, :] = v[:, :, 64:, :]
            cache_dic['enco_cache'][current['enco_layer_idx']]['v'][:, :, 0:64, :] = v[:, :, 0:64, :]

            self.cur_kv_len = self.cur_kv_len + app_kv_len
            available_kv_len = self.cur_kv_len + 64

            k = cache_dic['enco_cache'][current['enco_layer_idx']]['k']
            v = cache_dic['enco_cache'][current['enco_layer_idx']]['v']
        else:
            # Full refresh mode
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            
            cache_dic['enco_cache'][current['enco_layer_idx']]['k'].to(k)
            cache_dic['enco_cache'][current['enco_layer_idx']]['v'].to(k)
            
            if current['lazy_mar']:
                cache_dic['enco_cache'][current['enco_layer_idx']]['k'][:int(B/2), :self.num_heads, :N, :self.head_dim] = k[:int(B/2)]
                cache_dic['enco_cache'][current['enco_layer_idx']]['v'][:int(B/2), :self.num_heads, :N, :self.head_dim] = v[:int(B/2)]
            else:
                cache_dic['enco_cache'][current['enco_layer_idx']]['k'][:B, :self.num_heads, :N, :self.head_dim] = k[:B]
                cache_dic['enco_cache'][current['enco_layer_idx']]['v'][:B, :self.num_heads, :N, :self.head_dim] = v[:B]
            
            self.cur_kv_len = N - 64
            available_kv_len = self.cur_kv_len + 64
        
        # Compute attention
        q = q * self.scale
        attn = torch.matmul(q, k[:, :, :available_kv_len, :].transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v[:, :, :available_kv_len, :])
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        cache_dic['enco_cache'][current['enco_layer_idx']]['cur_kv_len'] = self.cur_kv_len
        self.cur_kv_len = 0
        return x


class DecoderAttention(nn.Module):
    """
    Multi-head self-attention for decoder with masked scatter caching support.
    
    Supports selective token update via masking mechanism for efficient decoding.
    
    Args:
        dim: Input feature dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        qk_norm: Whether to apply normalization to Q and K
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
        norm_layer: Normalization layer type
    """
    fused_attn: Final[bool]
    
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, current: dict, cache_dic: dict) -> torch.Tensor:
        """
        Forward pass with selective KV caching via masking.
        
        Args:
            x: Input tensor of shape (B, N, C)
            current: Dictionary containing current state and masks
            cache_dic: Dictionary containing cached KV pairs
            
        Returns:
            Output tensor of shape (B, N, C)
        """
        if current['lazy_mar'] and not current['is_force_fresh']:
            # Selective update mode
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            
            # Cache previous V for similarity computation at specific layer
            if current['layer_idx'] == 3:
                current['pre_cache_v'] = copy.deepcopy(cache_dic['cache'][current['layer_idx']]['v'])
            
            # Update cached KV selectively based on update mask
            if torch.all(current['update_mask'] == 1):
                # Full update
                cache_dic['cache'][current['layer_idx']]['k'] = k
                cache_dic['cache'][current['layer_idx']]['v'] = v
            else:
                # Masked scatter update
                mask_kv = current['update_mask'].unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim).bool()
                cache_dic['cache'][current['layer_idx']]['k'].masked_scatter_(mask_kv, k)
                k = cache_dic['cache'][current['layer_idx']]['k']
                cache_dic['cache'][current['layer_idx']]['v'].masked_scatter_(mask_kv, v)
                v = cache_dic['cache'][current['layer_idx']]['v']
        else:
            # Full refresh mode
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            
            # Initialize cache
            if current['lazy_mar']:
                cache_dic['cache'][current['layer_idx']]['k'] = k[:int(B/2)]
                cache_dic['cache'][current['layer_idx']]['v'] = v[:int(B/2)]
            else:
                cache_dic['cache'][current['layer_idx']]['k'] = k[:B]
                cache_dic['cache'][current['layer_idx']]['v'] = v[:B]
        
        # Compute attention
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncoderBlock(nn.Module):
    """
    Transformer encoder block with self-attention and MLP.
    
    Args:
        dim: Input feature dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension expansion ratio
        qkv_bias: Whether to use bias in QKV projection
        qk_norm: Whether to apply normalization to Q and K
        proj_drop: Projection dropout rate
        attn_drop: Attention dropout rate
        init_values: Initial value for LayerScale (None to disable)
        drop_path: DropPath rate
        act_layer: Activation layer type
        norm_layer: Normalization layer type
        mlp_layer: MLP layer type
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EncoderAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, current: dict, cache_dic: dict) -> torch.Tensor:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor
            current: Current state dictionary
            cache_dic: Cache dictionary
            
        Returns:
            Output tensor
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), current, cache_dic)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DecoderBlock(nn.Module):
    """
    Transformer decoder block with self-attention, MLP, and pruning support.
    
    Supports dynamic token pruning based on similarity scores for efficient generation.
    
    Args:
        dim: Input feature dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension expansion ratio
        qkv_bias: Whether to use bias in QKV projection
        qk_norm: Whether to apply normalization to Q and K
        proj_drop: Projection dropout rate
        attn_drop: Attention dropout rate
        init_values: Initial value for LayerScale (None to disable)
        drop_path: DropPath rate
        act_layer: Activation layer type
        norm_layer: Normalization layer type
        mlp_layer: MLP layer type
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DecoderAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, current: dict, cache_dic: dict) -> torch.Tensor:
        """
        Forward pass through decoder block with optional pruning.
        
        Args:
            x: Input tensor
            current: Current state dictionary containing pruning info
            cache_dic: Cache dictionary
            
        Returns:
            Output tensor
        """
        if current['lazy_mar'] and not current['is_force_fresh']:
            # Lazy MAR mode with pruning
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), current, cache_dic)))
            
            # Apply pruning at specific layer based on similarity
            if current['layer_idx'] == 3:
                x = self._prune_tokens(current, cache_dic, x)
            
            x_temp = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            x = x + x_temp
            
            # Restore full sequence at final layer
            if current['layer_idx'] == current['depth'] - 1:
                x = self._unprune_tokens(current, cache_dic, x)
        else:
            # Standard mode
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), current, cache_dic)))
            cache_dic['cache'][current['layer_idx']]['mlp'] = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            x = x + cache_dic['cache'][current['layer_idx']]['mlp']
        
        return x

    def _prune_tokens(self, current: dict, cache_dic: dict, x: torch.Tensor) -> torch.Tensor:
        """
        Prune tokens based on cosine similarity scores.
        
        Keeps tokens with high similarity changes and tokens to be predicted.
        
        Args:
            current: Current state dictionary
            cache_dic: Cache dictionary containing previous V values
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Pruned tensor with reduced sequence length
        """
        B, N, C = x.shape
        
        # Compute cosine similarity between current and previous V
        cos_sim = F.cosine_similarity(
            cache_dic['cache'][current['layer_idx']]['v'], 
            current['pre_cache_v'], 
            dim=-1
        )
        score = cos_sim.mean(dim=1).reshape(B, -1)
        
        # Mask out tokens that need prediction (keep them regardless of similarity)
        score[current['mask_to_pred_full']] = 0
        score[current['prev_mask_to_pred_full']] = 0
        
        # Select scores for updatable tokens only
        score = score[(current['update_mask']).nonzero(as_tuple=True)].reshape(B, -1)
        
        # Sort by similarity (ascending - keep tokens with low similarity = high change)
        _, inds = torch.sort(score, dim=-1, descending=False)
        
        # Determine number of tokens to keep based on retention schedule
        cur_ratio = RETAIN_RATIO_SCHEDULE[current['step']]
        fresh_num = torch.maximum(
            current['mask_to_pred_len'] + current['prev_mask_to_pred_len'] + 15,
            torch.tensor(int((inds.shape[1]) * cur_ratio)).to(current['mask_to_pred_len'])
        )
        inds = inds[:, :fresh_num]
        
        # Create next iteration mask
        next_mask = torch.zeros((B, N), device=x.device)
        next_mask = next_mask.scatter_(1, inds, 1)
        current['next_mask'] = next_mask
        
        # Update mask for next layers
        new_update_mask = torch.zeros_like(current['update_mask'], device=x.device).bool()
        new_update_mask.masked_scatter_(current['update_mask'], next_mask.bool())
        current['origi_update_mask'] = current['update_mask']
        current['update_mask'] = new_update_mask
        
        # Extract pruned tokens
        pruning_x = torch.masked_select(
            x, 
            next_mask.unsqueeze(-1).expand(-1, -1, C).bool()
        ).reshape(B, -1, C)
        
        return pruning_x

    def _unprune_tokens(self, current: dict, cache_dic: dict, pruning_x: torch.Tensor) -> torch.Tensor:
        """
        Restore full sequence from pruned tokens.
        
        Args:
            current: Current state dictionary
            cache_dic: Cache dictionary (unused but kept for interface consistency)
            pruning_x: Pruned tensor
            
        Returns:
            Full tensor with original sequence length
        """
        B, _, C = pruning_x.shape
        _, N = current['next_mask'].shape
        
        # Scatter pruned tokens back to full sequence
        new_x_full = torch.zeros((B, N, C), device=pruning_x.device)
        new_x_full.masked_scatter_(
            current['next_mask'].unsqueeze(-1).expand(-1, -1, C).bool(), 
            pruning_x
        )
        
        # Cleanup temporary state
        current['next_mask'] = None
        current['update_mask'] = current['origi_update_mask']
        current['origi_update_mask'] = None
        
        return new_x_full

