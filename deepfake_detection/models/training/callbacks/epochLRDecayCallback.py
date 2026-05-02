import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class EpochLrDecayCallback(TrainerCallback):
    """Apply manual epoch-based LR decay."""

    def __init__(self, every_n_epochs: int, gamma: float):
        self.every_n_epochs = every_n_epochs
        self.gamma = gamma

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.every_n_epochs or self.every_n_epochs <= 0:
            return control

        epoch = state.epoch
        if epoch is None:
            return control

        completed_epochs = int(round(epoch))
        if completed_epochs == 0 or (completed_epochs % self.every_n_epochs) != 0:
            return control

        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return control

        for param_group in optimizer.param_groups:
            param_group["lr"] *= self.gamma
        logger.info(
            "Applied manual LR decay at epoch %s with gamma=%s.",
            completed_epochs,
            self.gamma,
        )
        return control
