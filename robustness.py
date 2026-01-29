import argparse
import io
import os

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
import pandas as pd

from deepfake_detection.data.instance import ImageInstance
from deepfake_detection.utils.configuration import load_dataset, load_model
from deepfake_detection.evaluation.evaluator import Evaluator


def jpeg_compression(instance: ImageInstance, qf: float) -> ImageInstance:
    """
    Apply JPEG compression to an image instance with a given quality factor.
    """
    # Use a byte buffer to simulate saving as a JPEG
    buffer = io.BytesIO()
    instance.data.save(buffer, format="JPEG", quality=qf)

    # Reload the compressed image from the buffer
    buffer.seek(0)
    compressed_img = Image.open(buffer)

    # Update the instance and return it
    return ImageInstance(compressed_img)


def evaluate_robustness(model: str, dataset: str, output_dir: str):
    # TODO: Allow for multiple types of transformations

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset)

    # Load model
    print("Loading model...")
    model = load_model(model)

    # Define perturbation intensities
    intensities = np.arange(100, 10, -10)

    # Store results
    results = {}

    # Loop over perturbation intensities
    for idx, intensity in enumerate(intensities):
        print(f"{idx}/{len(intensities)}: Processing intensity {intensity}...")

        # Loop over dataset samples
        pbar = tqdm(dataset, desc='Making predictions...')
        predictions = []
        for batch in dataset.iter(batch_size=10):

            # Apply transformation
            batch = [jpeg_compression(i, qf=int(intensity)) if intensity < 100 else i for i in batch]

            # Make predictions
            predictions.extend(model.predict_batch(batch))

            # Update progress bar
            pbar.update(len(batch))

        # Evaluate predictions
        evaluator = Evaluator(dataset, predictions)
        metrics = evaluator.run(metrics=[balanced_accuracy_score,
                                         precision_score,
                                         recall_score],
                                label_type='authenticity'
                                )

        # Store results
        results[intensity] = {metric: val['all'] for metric, val in metrics.to_df().to_dict().items()}

    # Export results
    output_file = os.path.join(output_dir, f"{model.name}_robustness_results.csv")
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'JPEG quality factor'
    df.to_csv(output_file, index=True)
    print(f"Saved results in {output_file}!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        required=True,
                        help='Path to dataset config file.')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        help='Path to model config file.')
    parser.add_argument('-o',
                        '--output-dir',
                        default='results',
                        type=str,
                        required=False,
                        help='Path to directory where evaluation predictions should be saved.')
    evaluate_robustness(**vars(parser.parse_args()))
