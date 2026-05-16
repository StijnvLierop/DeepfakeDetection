import argparse
import copy
import logging
import os
import pydoc
from typing import Optional

import torch
import yaml
from torchvision.transforms import v2
from transformers import Trainer, TrainingArguments, set_seed

from deepfake_detection.data.datasets.torch import TorchDataset, TorchIterableDataset
from deepfake_detection.models import TrainableMixin
from deepfake_detection.utils.configuration import load_model, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Instantiate an optimizer from a config dict with 'class' and optional 'params' keys."""
    cls = pydoc.locate(cfg["class"])
    if cls is None:
        raise ValueError(f"Optimizer class not found: {cfg['class']}")
    return cls(model.parameters(), **cfg.get("params", {}))


def _build_augmentations(aug_cfgs: list) -> Optional[v2.Compose]:
    """Build a composed augmentation pipeline from a list of transform config dicts."""
    if not aug_cfgs:
        return None
    transforms = []
    for cfg in aug_cfgs:
        cls = pydoc.locate(cfg["class"])
        if cls is None:
            raise ValueError(f"Augmentation class not found: {cfg['class']}")
        transforms.append(cls(**cfg.get("params", {})))
    return v2.Compose(transforms)


def _build_callbacks(callback_cfgs: list) -> list:
    """Instantiate Trainer callbacks from config dicts."""
    callbacks = []
    for cfg in callback_cfgs:
        cls = pydoc.locate(cfg["class"])
        if cls is None:
            raise ValueError(f"Callback class not found: {cfg['class']}")
        callbacks.append(cls(**cfg.get("params", {})))
    return callbacks


class _TransformWithParams:
    """Pickle-safe transform wrapper that applies keyword overrides."""

    def __init__(self, base_transform, params: dict):
        self.base_transform = base_transform
        self.params = params

    def __call__(self, instance):
        return self.base_transform(instance, **self.params)


def _build_transform(base_transform, params: Optional[dict]):
    """Return a transform callable, optionally applying keyword overrides."""
    if not params:
        return base_transform
    return _TransformWithParams(base_transform, params)


class _ChainedTransform:
    """Apply pre-augmentations on PIL, then base transform, then optional post-augmentations on tensor."""

    def __init__(self, base_transform, pre_augmentations=None, post_augmentations=None):
        self.base_transform = base_transform
        self.pre_augmentations = pre_augmentations
        self.post_augmentations = post_augmentations

    def __call__(self, instance):
        # Pre-augmentations run on raw PIL image before base_transform.
        augmented = instance.data
        if self.pre_augmentations is not None:
            augmented = self.pre_augmentations(augmented)

        # A shallow copy is used so the dataset's cached instance is not mutated.
        proxy = copy.copy(instance)
        proxy.data = augmented
        transformed = self.base_transform(proxy)

        # Post-augmentations run on transformed tensors.
        if self.post_augmentations is not None:
            transformed = self.post_augmentations(transformed)
        return transformed


def _build_torch_dataset(dataset_cfg: dict, transform, label: str, pos_label: str):
    """Wrap a dataset config in the appropriate Torch dataset class based on its type."""
    dataset = load_dataset(dataset_cfg)
    try:
        len(dataset)
        return TorchDataset(dataset, transform, label, pos_label)
    except TypeError:
        return TorchIterableDataset(dataset, transform, label, pos_label)


def train(
    config_path: str,
    train_dataset_config: str,
    val_dataset_config: Optional[str] = None,
) -> None:
    """Run the full training pipeline from a YAML config and dataset config paths."""

    # Load the main training config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    training_cfg = config["training"]
    label: str = training_cfg.get("label")
    pos_label: str = training_cfg.get("pos_label")

    if "seed" in training_cfg:
        set_seed(int(training_cfg["seed"]))

    # Instantiate and validate the model
    model = load_model(config["model"])
    if not isinstance(model, TrainableMixin):
        raise ValueError(
            f"Model '{model.name}' does not support training (not a TrainableMixin)."
        )

    # Build the optimizer from config
    optimizer = _build_optimizer(model, training_cfg["optimizer"])

    # Compose the training transform: model preprocessing + optional augmentations
    train_transform_params = training_cfg.get("train_transform_params")
    eval_transform_params = training_cfg.get("eval_transform_params")
    base_train_transform = _build_transform(
        model.transform_input, train_transform_params
    )
    base_eval_transform = _build_transform(model.transform_input, eval_transform_params)

    pre_augmentations = _build_augmentations(training_cfg.get("augmentations", []))
    post_augmentations = _build_augmentations(
        training_cfg.get("post_augmentations", [])
    )
    if pre_augmentations is not None or post_augmentations is not None:
        train_transform = _ChainedTransform(
            base_train_transform,
            pre_augmentations=pre_augmentations,
            post_augmentations=post_augmentations,
        )
    else:
        train_transform = base_train_transform

    # Build train and optional validation datasets
    train_dataset = _build_torch_dataset(
        train_dataset_config, train_transform, label, pos_label
    )
    val_dataset = (
        _build_torch_dataset(val_dataset_config, base_eval_transform, label, pos_label)
        if val_dataset_config
        else None
    )

    # Configure MLflow tracking if specified in the config
    report_to = "none"
    if "mlflow" in training_cfg:
        mlflow_cfg = training_cfg["mlflow"]
        tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
        # Normalize bare .db paths to a proper SQLite URI
        if tracking_uri.endswith(".db") and not tracking_uri.startswith("sqlite:"):
            tracking_uri = f"sqlite:///{tracking_uri}"
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_cfg.get(
            "experiment_name", "deepfake_training"
        )
        if "run_name" in mlflow_cfg:
            os.environ["MLFLOW_RUN_NAME"] = mlflow_cfg["run_name"]
        report_to = "mlflow"

    # Build HuggingFace TrainingArguments from config values
    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"]
        if val_dataset
        else None,
        dataloader_num_workers=training_cfg["dataloader_num_workers"],
        dataloader_pin_memory=training_cfg.get("dataloader_pin_memory", True),
        dataloader_persistent_workers=training_cfg.get("dataloader_persistent_workers", False),
        dataloader_drop_last=training_cfg.get("dataloader_drop_last", False),
        fp16=training_cfg.get("fp16", False),
        bf16=training_cfg.get("bf16", False),
        torch_compile=training_cfg.get("torch_compile", False),
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_steps=training_cfg["warmup_steps"],
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        load_best_model_at_end=bool(val_dataset),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
        label_names=getattr(model, "label_names", None),
        report_to=report_to,
        remove_unused_columns=False,
        logging_steps=10,
        logging_first_step=True,
    )

    # Assemble and run the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
        data_collator=model.data_collator,
    )

    # Add configured callbacks
    callback_cfgs = list(training_cfg.get("callbacks", []))
    for callback in _build_callbacks(callback_cfgs):
        trainer.add_callback(callback)

    # Train the model
    trainer.train()

    # After training, the Trainer restores the best checkpoint when
    # load_best_model_at_end=True, so model weights here are the best seen.
    model_label = "best" if val_dataset else "final"
    save_path = os.path.join(training_cfg["output_dir"], "model.pth")
    torch.save(model.state_dict(), save_path)
    logger.info("Training complete. %s model saved to %s", model_label, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a deepfake detection model from a YAML configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "-d",
        "--train-dataset",
        type=str,
        required=True,
        help="Path to training dataset configuration YAML file.",
    )
    parser.add_argument(
        "-v",
        "--val-dataset",
        type=str,
        required=False,
        default=None,
        help="Path to validation dataset configuration YAML file.",
    )
    args = parser.parse_args()
    train(args.config, args.train_dataset, args.val_dataset)
