"""
Factory for creating DiffLoss instances.

This module provides a factory function to create the appropriate DiffLoss
implementation based on whether caching (LazyMAR optimization) is enabled.
"""

from models.diffloss import DiffLoss
from models.diffloss_cached import CachedDiffLoss


def create_diffloss(
    target_channels: int,
    z_channels: int,
    width: int,
    depth: int,
    num_sampling_steps: str,
    grad_checkpointing: bool = False,
    enable_cache: bool = False,
) -> DiffLoss:
    """
    Factory function to create the appropriate DiffLoss implementation.
    
    Args:
        target_channels: Number of target channels (token embedding dimension)
        z_channels: Number of conditioning channels (decoder embedding dimension)
        width: Width of the diffusion MLP
        depth: Depth of the diffusion MLP (number of residual blocks)
        num_sampling_steps: Number of sampling steps for generation
        grad_checkpointing: Whether to use gradient checkpointing
        enable_cache: If True, creates CachedDiffLoss for LazyMAR acceleration.
                     If False, creates standard DiffLoss.
    
    Returns:
        An instance of DiffLoss or CachedDiffLoss based on enable_cache parameter.
    
    Example:
        >>> # For standard MAR
        >>> diffloss = create_diffloss(
        ...     target_channels=16,
        ...     z_channels=1280,
        ...     width=1536,
        ...     depth=12,
        ...     num_sampling_steps="100",
        ...     enable_cache=False
        ... )
        
        >>> # For LazyMAR with caching
        >>> diffloss = create_diffloss(
        ...     target_channels=16,
        ...     z_channels=1280,
        ...     width=1536,
        ...     depth=12,
        ...     num_sampling_steps="100",
        ...     enable_cache=True
        ... )
    """
    if enable_cache:
        return CachedDiffLoss(
            target_channels=target_channels,
            z_channels=z_channels,
            width=width,
            depth=depth,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
    else:
        return DiffLoss(
            target_channels=target_channels,
            z_channels=z_channels,
            width=width,
            depth=depth,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )

