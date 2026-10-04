"""
Self-correction training utilities for StreamVLN.

This module previously implemented loss masking; it now keeps a compatibility trainer wrapper
that simply delegates to the base LLaVATrainer.
"""

from .trainer import SelfCorrectLLaVATrainer, create_selfcorrect_trainer

__all__ = [
    'SelfCorrectLLaVATrainer',
    'create_selfcorrect_trainer',
]
