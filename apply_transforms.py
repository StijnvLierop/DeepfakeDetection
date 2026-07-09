import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from deepfake_detection.data.instance import FileImageInstance
from deepfake_detection.utils.configuration import load_dataset, load_transforms


def _apply_to_image(pil_img: Image.Image, transforms, output_dir: Path, stem: str):
    """
    Apply transforms to an image and save results.

    :param pil_img: The PIL image to apply transforms to.
    :param transforms: List of transforms to apply.
    :param output_dir: Directory to save transformed images.
    :param stem: Stem for output filenames.
    """
    array = np.array(pil_img.convert("RGB"))
    for transform in transforms:
        result = transform.apply(array)
        transform_dir = output_dir / transform.name
        transform_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.astype(np.uint8)).save(transform_dir / f"{stem}.png")


def apply_transforms(
    transforms: str,
    output: str,
    image: Optional[str] = None,
    dataset: Optional[str] = None,
):
    """
    Apply analysis transforms to an image or dataset and save results.

    :param transforms: Path to transforms config YAML.
    :param output: Output directory.
    :param image: Path to a single input image.
    :param dataset: Path to a dataset config YAML.
    """
    # Input guards
    if not image and not dataset:
        raise ValueError("Provide either --image or --dataset.")
    if image and dataset:
        raise ValueError("Provide either --image or --dataset, not both.")

    # Set output dir
    output_dir = Path(output)

    # Load transforms
    loaded_transforms = load_transforms(transforms)

    # Load image (if image configured) and apply transforms to image
    if image:
        img_path = Path(image)
        pil_img = Image.open(img_path)
        _apply_to_image(pil_img, loaded_transforms, output_dir, img_path.stem)
        print(f"Saved results to {output_dir}")
        return

    # Load dataset (if dataset configured) and apply transforms to each image in dataset
    ds = load_dataset(dataset)
    for i, instance in enumerate(ds):
        pil_img = instance.data if hasattr(instance, "data") else None
        if pil_img is None:
            print(f"Skipping instance {i}: no image data.")
            continue
        stem = instance.path.stem if isinstance(instance, FileImageInstance) else str(i)
        _apply_to_image(pil_img, loaded_transforms, output_dir, stem)

    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply analysis transforms to an image or dataset and save results."
    )
    parser.add_argument(
        "-t", "--transforms", required=True, help="Path to transforms config YAML."
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--image", help="Path to a single input image.")
    source.add_argument("-d", "--dataset", help="Path to a dataset config YAML.")
    apply_transforms(**vars(parser.parse_args()))
