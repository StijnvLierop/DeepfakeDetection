import argparse
import logging
from pathlib import Path

import yaml

from deepfake_detection.data.instance import FileVideoInstance
from deepfake_detection.models.detection.rppg import rPPGLSTM
from deepfake_detection.utils.configuration import load_dataset, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_rppg_features(
    model_config: str,
    dataset_config: str,
    output_dir: str,
    skip_existing: bool = True,
) -> None:
    """
    Extract rPPG features from every video in a dataset and save them as .npy files.

    Saved files use the same naming convention as the model's feature cache
    (``{hash(video_path)}.npy``), so the output directory can be passed directly
    as ``feature_cache_dir`` in the training config to avoid re-extraction.

    :param model_config: Path to the model YAML config (used for extraction params only;
                         model weights are not loaded).
    :param dataset_config: Path to the dataset YAML config.
    :param output_dir: Directory where .npy feature files will be written.
    :param skip_existing: Skip videos whose .npy file already exists (default True).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model config without weights to get extraction parameters.
    with open(model_config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", cfg)
    # Force load_model=False — we only need the feature extractor, not trained weights.
    model_cfg.setdefault("params", {})["load_model"] = False
    model = load_model(model_cfg)

    if not isinstance(model, rPPGLSTM):
        raise ValueError(
            f"extract_rppg_features only supports rPPGLSTM, got {type(model).__name__}"
        )

    # Point the model at our output directory so _cache_path produces the right paths.
    model.feature_cache_dir = output_path

    dataset = load_dataset(dataset_config)

    total = skipped = extracted = errors = 0
    for instance in dataset:
        total += 1

        if not isinstance(instance, FileVideoInstance):
            logger.warning("Skipping non-video instance: %s", type(instance).__name__)
            continue

        cache = model._cache_path(instance)
        if skip_existing and cache is not None and cache.exists():
            skipped += 1
            continue

        try:
            features = model.extract_features(instance.data)
            import numpy as np
            np.save(cache, features)
            extracted += 1
            logger.info("[%d] Extracted %s → %s", total, instance.path.name, cache.name)
        except Exception as exc:
            errors += 1
            logger.error("Failed to extract features for %s: %s", instance.path, exc)

    logger.info(
        "Done. %d total | %d extracted | %d skipped | %d errors",
        total, extracted, skipped, errors,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract rPPG features from a video dataset and save as .npy files."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the rPPGLSTM model YAML config (extraction params only; weights not loaded).",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Path to the dataset YAML config.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Directory to write .npy feature files into.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        default=False,
        help="Re-extract even if a .npy file already exists (default: skip existing).",
    )
    args = parser.parse_args()
    extract_rppg_features(
        model_config=args.config,
        dataset_config=args.dataset,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip,
    )
