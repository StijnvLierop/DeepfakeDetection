import argparse
import logging
import os.path
from pathlib import Path
from typing import List, Optional

from datasets import tqdm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score, balanced_accuracy_score,
)

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.evaluation.evaluator import Evaluator
from deepfake_detection.models import Model, Prediction
from deepfake_detection.utils.configuration import load_model, load_dataset
from deepfake_detection.utils.io import (
    read_predictions_from_file,
    write_predictions_to_file,
)


# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)


def evaluate(dataset_config: str, model_config: str, output_dir: str, predictions_file: Optional[str] = None) -> None:
    """
    Evaluates a model on specified datasets and generates metrics and visualizations.

    The function loads datasets specified in a dataset configuration file and a model from a model configuration file.
    It evaluates the model performance on each dataset, calculates metrics, and generates plots or other evaluative
    outputs. The results are stored in the specified output directory.

    :param dataset_config: File path to the dataset configuration.
    :param model_config: File path to the model configuration. Should contain the configuration for a single model class.
    :param output_dir: Directory path where to store model predictions and evaluation results.
    :param predictions_file: Optional file path to a file containing model predictions.
                             If specified, the function will use these predictions instead of generating new ones.
    """
    # Load datasets
    dataset = load_dataset(dataset_config)

    # Load model
    model = load_model(model_config)

    # Log evaluation information
    logging.info(
        f"Evaluating {model.name} model on dataset: {dataset.dataset_name}"
    )

    logging.info(f"Evaluating {dataset.dataset_name}...")

    # Make or retrieve cached predictions
    if predictions_file:
        logging.info(f"Reading predictions from {predictions_file}")
        predictions = read_predictions_from_file(predictions_file)
    else:
        # Make predictions
        predictions = []
        for instance in tqdm(
                dataset, desc=f"Making predictions for {dataset.dataset_name}", total=len(dataset)
        ):
            prediction = model.predict(instance)
            predictions.append(prediction)

        # Write predictions to a file
        predictions_file = os.path.join(output_dir, f"{model.name}_{dataset.name}.json")
        write_predictions_to_file(predictions, Path(predictions_file))
        logging.info(f"Saved predictions to {predictions_file}")

    # Calculate metrics / make plots and write to output dir
    evaluate_model_on_dataset(dataset, model, predictions, output_dir)


def evaluate_model_on_dataset(
    dataset: Dataset, model: Model, predictions: List[Prediction], output_dir: str
) -> None:
    """
    Evaluates the performance of a prediction model on a given dataset by computing classification metrics,
    generating outputs, and saving the metrics to a file in the specified directory.

    :param dataset: The dataset object containing the ground truth data.
    :param model: The model to evaluate.
    :param predictions: A list of prediction objects corresponding to the dataset.
    :param output_dir: The directory path where the evaluation metrics are to be saved.
    """
    # Make evaluator
    evaluator = Evaluator(list(dataset), predictions)

    # Get overall evaluation results
    overall_results = evaluator.run(
        [
            balanced_accuracy_score,
            accuracy_score,
            average_precision_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        ],
        label_type="authenticity_label",
    )

    # Get per-subset results
    per_generator_results = evaluator.run(
        [balanced_accuracy_score, accuracy_score, average_precision_score, roc_auc_score],
        label_type="authenticity_label",
        group_by="source_label",
    )

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics to file
    overall_results.to_df().to_csv(
        os.path.join(output_dir, f"{dataset.dataset_name}_{model.name}_overall_metrics.csv")
    )
    per_generator_results.to_df().to_csv(
        os.path.join(output_dir, f"{dataset.dataset_name}_{model.name}_subset_metrics.csv")
    )

    logging.info(
        f"Exported evaluation results to '{dataset.dataset_name}_{model.name}_overall_metrics.csv'"
        f" and '{dataset.dataset_name}_{model.name}_subset_metrics.csv'"
    )


if __name__ == "__main__":
    # Define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset-config",
        type=str,
        required=True,
        help="Path to dataset config file.",
    )
    parser.add_argument(
        "-m",
        "--model-config",
        type=str,
        required=True,
        help="Path to model config file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="results",
        type=str,
        required=False,
        help="Path to directory where evaluation predictions should be saved.",
    )
    parser.add_argument(
        "-p",
        "--predictions-file",
        type=str,
        required=False,
        help="Path to a file containing model predictions.",
    )
    args = vars(parser.parse_args())

    # Run evaluation
    evaluate(**args)
