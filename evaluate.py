import argparse
import logging
import os.path
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from datasets import tqdm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
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


def evaluate(
    dataset_config: str,
    model_config: str,
    output_dir: str,
    label: str,
    device: str,
    predictions_file: Optional[str] = None,
    subset_labels: Optional[List[str]] = None,
    neg_label: Optional[str] = None,
) -> None:
    """
    Evaluates a model on specified datasets and generates metrics and visualizations.

    The function loads datasets specified in a dataset configuration file and a model from a model configuration file.
    It evaluates the model performance on each dataset, calculates metrics, and generates plots or other evaluative
    outputs. The results are stored in the specified output directory.

    :param dataset_config: File path to the dataset configuration.
    :param model_config: File path to the model configuration. Should contain the configuration for a single model class.
    :param output_dir: Directory path where to store model predictions and evaluation results.
    :param label: Label to use for calculating metrics.
    :param predictions_file: Optional file path to a file containing model predictions.
                             If specified, the function will use these predictions instead of generating new ones.
    :param subset_labels: Optional list of subset labels to report metrics for per subset.
                          If left empty, metrics are only calculated for the full dataset.
    :param neg_label: Optional negative label to use for calculating subset metrics that need two classes.
                      If not provided, these metrics will not be included in the evaluation.
    :param device: Device to use for model inference. Defaults to 'cuda'.
    """
    # Load datasets
    dataset = load_dataset(dataset_config)

    # Load model
    model = load_model(model_config)

    # Transfer model to device (if applicable)
    if isinstance(model, torch.nn.Module):
        model = model.to(device)

    # Log evaluation information
    logging.info(f"Evaluating {model.name} model on dataset: {dataset.dataset_name}")

    logging.info(f"Evaluating {dataset.dataset_name}...")

    # Make or retrieve cached predictions
    if predictions_file:
        logging.info(f"Reading predictions from {predictions_file}")
        predictions = read_predictions_from_file(predictions_file)
        instances = list(dataset)
    else:
        # Make predictions, skipping corrupt files
        instances = []
        predictions = []
        for instance in tqdm(
            dataset,
            desc=f"Making predictions for {dataset.dataset_name}",
            total=len(dataset),
        ):
            try:
                prediction = model.predict(instance)
            except (SyntaxError, OSError) as e:
                logging.warning(f"Skipping corrupt file {getattr(instance, 'path', '?')}: {e}")
                continue
            instances.append(instance)
            predictions.append(prediction)

        # Write predictions to a file
        predictions_file = os.path.join(
            output_dir, f"{model.name}_{dataset.dataset_name}.json"
        )
        write_predictions_to_file(predictions, Path(predictions_file))
        logging.info(f"Saved predictions to {predictions_file}")

    # Calculate metrics / make plots and write to output dir
    evaluate_model_on_dataset(
        dataset, instances, model, predictions, output_dir, label, subset_labels, neg_label
    )


def evaluate_model_on_dataset(
    dataset: Dataset,
    instances: list,
    model: Model,
    predictions: List[Prediction],
    output_dir: str,
    label: str,
    subset_labels: Optional[List[str]] = None,
    neg_label: Optional[str] = None,
) -> None:
    """
    Evaluates the performance of a prediction model on a given dataset by computing classification metrics,
    generating outputs, and saving the metrics to a file in the specified directory.

    :param dataset: The dataset object containing the ground truth data.
    :param model: The model to evaluate.
    :param predictions: A list of prediction objects corresponding to the dataset.
    :param output_dir: The directory path where the evaluation metrics are to be saved.
    :param label: Label to use for calculating metrics.
    :param subset_labels: Optional list of subset labels to report metrics for per subset.
                          If left empty, metrics are only calculated for the full dataset.
    :param neg_label: Optional negative label to use for calculating subset metrics that need two classes.
                      If not provided, these metrics will not be included in the evaluation.
    """
    # Make evaluator
    evaluator = Evaluator(instances, predictions)

    # Define metrics
    metrics = [
        balanced_accuracy_score,
        accuracy_score,
        average_precision_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    ]

    # Get overall evaluation results
    overall_results = evaluator.run(metrics, label_type=label)
    overall_df = overall_results.to_df()
    overall_df["evaluation"] = ["all"]
    overall_df["label"] = overall_df.index
    combined_df = overall_df

    # Get per-subset results
    if subset_labels:
        for subset_label in subset_labels:
            per_subset_results = evaluator.run(
                metrics,
                label_type=label,
                group_by=subset_label,
                negative_class_label=neg_label,
            )
            per_subset_results = per_subset_results.to_df()
            per_subset_results["evaluation"] = subset_label
            per_subset_results["label"] = per_subset_results.index
            combined_df = pd.concat(
                [overall_df, per_subset_results], axis=0, ignore_index=True
            )

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics to file
    combined_df.to_csv(
        os.path.join(output_dir, f"{dataset.dataset_name}_{model.name}_metrics.csv"),
        index=False,
    )

    logging.info(
        f"Exported evaluation results to '{dataset.dataset_name}_{model.name}_metrics.csv'"
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
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        required=True,
        help="Label to use for calculating metrics.",
    )
    parser.add_argument(
        "-s",
        "--subset-labels",
        type=str,
        nargs="+",
        required=False,
        help="Optional list of subset labels to calculate metrics for separately."
        "If left empty, metrics are only calculated for the full dataset.",
    )
    parser.add_argument(
        "-n",
        "--neg-label",
        type=str,
        required=False,
        help="Optional negative label to use for calculating subset metrics that need two classes."
        "If not provided, these metrics will not be included in the evaluation.",
    )
    parser.add_argument(
        "-dev",
        "--device",
        type=str,
        required=False,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for model inference. Defaults to 'cuda' if available, otherwise 'cpu'.",
    )
    args = vars(parser.parse_args())

    # Run evaluation
    evaluate(**args)
