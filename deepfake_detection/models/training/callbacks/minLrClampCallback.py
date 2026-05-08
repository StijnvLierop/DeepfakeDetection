import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class MinLrClampCallback(TrainerCallback):
    """Clamp optimizer learning rates to a configured minimum value."""

    def __init__(self, min_lr: float):
        self.min_lr = float(min_lr)

    def on_step_end(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return control

        changed = False
        for param_group in optimizer.param_groups:
            old_lr = float(param_group.get("lr", 0.0))
            if old_lr < self.min_lr:
                param_group["lr"] = self.min_lr
                changed = True

        if (
            changed
            and state.global_step > 0
            and state.global_step % args.logging_steps == 0
        ):
            logger.info(
                "Clamped learning rate to min_lr=%s at step=%s",
                self.min_lr,
                state.global_step,
            )
        return control
