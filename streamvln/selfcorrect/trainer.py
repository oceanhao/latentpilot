"""
Thin compatibility trainer: delegates to base LLaVATrainer with no loss masking logic.
"""

from llava.train.llava_trainer import LLaVATrainer


class SelfCorrectLLaVATrainer(LLaVATrainer):
    """Compatibility trainer without self-correction behavior."""

    def compute_loss(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


# Convenience function to create trainer

def create_selfcorrect_trainer(model, tokenizer, training_args, **data_module):
    """Create a trainer (kept for API compatibility)."""
    return SelfCorrectLLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
