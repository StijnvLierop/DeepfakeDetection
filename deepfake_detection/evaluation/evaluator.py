from typing import List, Callable, Union, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from deepfake_detection.data.instance import Instance
from deepfake_detection.evaluation.utils import to_arrays, get_labels
from deepfake_detection.models import Prediction


# Metrics that definitely need probabilities (y_score), not hard labels (y_pred)
PROBABILITY_METRICS = {
    roc_auc_score,
    average_precision_score,
}


class EvaluationResult:
    """
    A helper class to store the output of the Evaluator.
    """

    def __init__(self, scores: Mapping[str, Mapping[str, float]]):
        """
        :param scores: A mapping of metrics to mappings of labels to metric scores.
        """
        self.scores = scores

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.scores).T

    def __repr__(self):
        return str(self.to_df())


class Evaluator:
    """
    Class that can be used to run multiple evaluations on a series of instances and corresponding predictions.
    """

    def __init__(self, instances: List[Instance], predictions: List[Prediction]):
        # Ensure that instances and predictions have the same length
        if len(instances) != len(predictions):
            raise ValueError(
                f"Mismatch: {len(instances)} instances vs {len(predictions)} predictions."
            )

        self.instances = instances
        self.predictions = predictions

        # Create a dataframe index to quickly evaluate based on various metadata fields
        self._metadata_index = self._build_metadata_index()

    def _build_metadata_index(self) -> pd.DataFrame:
        """
        Extracts metadata into a dataframe for fast grouping/filtering.
        """
        rows = []
        for idx, (inst, pred) in enumerate(zip(self.instances, self.predictions)):
            row = {"_idx": idx}

            # Extract annotation fields
            if inst.annotation:
                row.update(
                    {
                        label: inst.annotation.get_label(label) for label in inst.annotation.labels
                    }
                )

            # Extract instance metadata
            if inst.meta:
                row.update(inst.meta)

            # Extract prediction metadata
            if pred.meta:
                row.update(pred.meta)

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, metrics: List[Callable]
    ) -> Mapping[str, float]:
        """
        Runs all given metrics on a series of ground-truth labels (y_true) and corresponding predictions (y_pred).
        """
        return {m.__name__: m(y_true, y_pred) for m in metrics}

    @staticmethod
    def _is_probability_metric(metric: Callable) -> bool:
        """
        Heuristic to determine if a metric needs probabilities or hard labels.
        """
        # Check against known sklearn metrics
        if metric in PROBABILITY_METRICS:
            return True

        # Check function name for common probability keywords (for custom metrics)
        name = getattr(metric, "__name__", "").lower()
        if any(x in name for x in ["auc", "probability", "log_loss", "brier"]):
            return True

        return False

    def _calculate_one_vs_rest(
        self,
        instances: List[Instance],
        predictions: List[Prediction],
        label: str,
        label_type: str,
        metrics: List[Callable],
        threshold: float,
    ) -> Mapping[str, float]:
        """
        Calculates metrics for a single class vs others.

        :param instances: Instances with ground-truth labels.
        :param predictions: Predictions with classifications.
        :param label: Label to calculate one vs rest metric for.
        :param label_type: The label type in 'Annotation' to use for computing the metric(s).
        :param metrics: Metrics to calculate.
        :param threshold: Threshold to use for converting confidence scores to predicted labels.
        """
        # Calculate hard and soft labels
        y_true, y_pred = to_arrays(
            instances, predictions, label, label_type, threshold, binary=True
        )
        _, y_prob = to_arrays(
            instances, predictions, label, label_type, threshold, binary=False
        )

        # Calculate metrics
        return {
            m.__name__: m(y_true, y_prob)
            if self._is_probability_metric(m)
            else m(y_true, y_pred)
            for m in metrics
        }

    def _calculate_macro_average(
        self,
        instances: List[Instance],
        predictions: List[Prediction],
        label_type: str,
        metrics: List[Callable],
        threshold: float,
    ) -> Mapping[str, float]:
        """
        Calculates metrics for every label and averages them.

        :param instances: Instances with ground-truth labels.
        :param predictions: Predictions with classifications.
        :param label_type: The label type in 'Annotation' to use for computing the metric(s).
        :param metrics: Metrics to calculate.
        :param threshold: Threshold to use for converting confidence scores to predicted labels.
        """
        # Get labels
        labels = get_labels(instances, predictions, label_type)

        # Calculate metric for each class
        all_class_results = [
            self._calculate_one_vs_rest(
                instances, predictions, lbl, label_type, metrics, threshold
            )
            for lbl in labels
        ]
        # Return dictionary of results
        return pd.DataFrame(all_class_results).mean().to_dict()

    def run(
        self,
        metrics: Union[Callable, List[Callable]],
        label_type: str,
        threshold: float = 0.5,
        target_class: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Executes one or multiple given evaluation functions across the dataset or per group.

        :param metrics: A function or list of functions.
                        Each must accept (List[Instance], List[Prediction]).
        :param label_type: The label type in 'Annotation' to use for computing the metric(s).
        :param threshold: An optional threshold to use for converting confidence scores to predicted labels.
        :param target_class: If specified, will only return scores for this class label.
                             Otherwise, will return macro average of all labels.
        :param group_by: The metadata key to group by (e.g., 'source', 'model_version').
        """
        # Transform metrics to list if only one provided
        if not isinstance(metrics, list):
            metrics = [metrics]

        # Prepare Slices
        if group_by:
            # Raise error if group key not found
            if group_by not in self._metadata_index.columns:
                raise ValueError(f"Key '{group_by}' not found.")
            grouped_indices = self._metadata_index.groupby(group_by).indices
        else:
            # If no group_by, we have one slice called "overall"
            grouped_indices = {"overall": self._metadata_index.index.tolist()}

        # Create dictionary to store results
        final_results = {}

        # Iterate through slices and calculate metrics for each slice
        for slice_name, indices in grouped_indices.items():
            slice_inst = [self.instances[i] for i in indices]
            slice_pred = [self.predictions[i] for i in indices]

            # If a target class is provided, calculate one vs rest score
            if target_class:
                scores = self._calculate_one_vs_rest(
                    slice_inst, slice_pred, target_class, label_type, metrics, threshold
                )
            # Otherwise calculate macro average of all classes
            else:
                scores = self._calculate_macro_average(
                    slice_inst, slice_pred, label_type, metrics, threshold
                )

            # Store scores in final results
            final_results[str(slice_name)] = scores

        return EvaluationResult(final_results)
