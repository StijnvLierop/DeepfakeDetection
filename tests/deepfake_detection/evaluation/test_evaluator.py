import pytest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)

from deepfake_detection.evaluation.evaluator import Evaluator, EvaluationResult


def test_init(instances, authenticity_predictions):
    evaluator = Evaluator(instances, authenticity_predictions)
    assert len(evaluator.instances) == len(instances)
    assert len(evaluator.predictions) == len(authenticity_predictions)


def test_init_value_error(instances, authenticity_predictions):
    with pytest.raises(ValueError):
        Evaluator(instances, authenticity_predictions[:-1])


def test_build_metadata_index(instances, authenticity_predictions):
    evaluator = Evaluator(instances, authenticity_predictions)
    df = evaluator._build_metadata_index()
    assert df.shape[0] == len(instances)


def test_run_single_metric(instances, authenticity_predictions):
    evaluator = Evaluator(instances, authenticity_predictions)
    result = evaluator.run(accuracy_score, label_type="authenticity")
    assert isinstance(result, EvaluationResult)
    assert "overall" in result.scores


def test_run_multiple_metrics(instances, authenticity_predictions):
    evaluator = Evaluator(instances, authenticity_predictions)
    result = evaluator.run(
        [accuracy_score, average_precision_score], label_type="authenticity"
    )
    assert isinstance(result, EvaluationResult)
    assert "overall" in result.scores


def test_run_group_by(instances, authenticity_predictions):
    evaluator = Evaluator(instances, authenticity_predictions)
    result = evaluator.run(
        accuracy_score, group_by="authenticity", label_type="authenticity"
    )
    assert isinstance(result, EvaluationResult)
    for group_name in evaluator._metadata_index["authenticity"].unique():
        assert str(group_name) in result.scores


def test_run_group_by_value_error(instances, authenticity_predictions):
    evaluator = Evaluator(instances, authenticity_predictions)
    with pytest.raises(ValueError):
        evaluator.run(
            accuracy_score, group_by="nonexistent", label_type="authenticity"
        )


def test_accuracy_authenticity(instances, authenticity_predictions):
    evaluator = Evaluator(instances, authenticity_predictions)
    result = evaluator.run(accuracy_score, label_type="authenticity")
    assert pytest.approx(result.scores["overall"]["accuracy_score"]) == 5 / 8


def test_precision_authenticity_specified_label(instances, authenticity_predictions):
    evaluator = Evaluator(instances, authenticity_predictions)
    result = evaluator.run(
        precision_score, target_class="fake", label_type="authenticity"
    )
    assert pytest.approx(result.scores["overall"]["precision_score"]) == 2 / 3
    result = evaluator.run(
        precision_score, target_class="real", label_type="authenticity"
    )
    assert pytest.approx(result.scores["overall"]["precision_score"]) == 3 / 5


def test_recall_authenticity_specified_label(instances, authenticity_predictions):
    evaluator = Evaluator(instances, authenticity_predictions)
    result = evaluator.run(
        recall_score, target_class="fake", label_type="authenticity"
    )
    assert pytest.approx(result.scores["overall"]["recall_score"]) == 2 / 4
    result = evaluator.run(
        recall_score, target_class="real", label_type="authenticity"
    )
    assert pytest.approx(result.scores["overall"]["recall_score"]) == 3 / 4


def test_accuracy_grouped_by_source(instances, authenticity_predictions):
    # Add custom metadata to instances to test grouping by something other than labels
    for i, inst in enumerate(instances):
        inst.meta = {"quality": "high" if i < 4 else "low"}

    # Run evaluation
    evaluator = Evaluator(instances, authenticity_predictions)
    result = evaluator.run(
        accuracy_score, group_by="quality", label_type="authenticity"
    )

    # High quality (indices 0-3) are all actually "real"
    # Low quality (indices 4-7) are all actually "fake"
    # In Low quality, predictions for 'fake' were: 0.7, 0.2, 0.8, 0.3.
    # Two are >= 0.5 (hits), two are not. Accuracy = 0.5
    assert pytest.approx(result.scores["low"]["accuracy_score"]) == 2 / 4


def test_mixed_metrics_automation(instances, authenticity_predictions):
    evaluator = Evaluator(instances, authenticity_predictions)

    # Run both a threshold-based metric and a probability-based metric
    result = evaluator.run(
        metrics=[accuracy_score, roc_auc_score], label_type="authenticity"
    )

    # Check that both metrics exist in the output
    assert "accuracy_score" in result.scores["overall"]
    assert "roc_auc_score" in result.scores["overall"]
    # ROC AUC for 'real' (labels: 1, 0, 1, 1 | scores: 0.8, 0.4, 1.0, 0.8) should be 1.0
    assert result.scores["overall"]["roc_auc_score"] > 0


def test_accuracy_with_different_thresholds(instances, authenticity_predictions):
    evaluator = Evaluator(instances, authenticity_predictions)
    res_08 = evaluator.run(
        accuracy_score, threshold=0.8, label_type="authenticity"
    )
    res_01 = evaluator.run(
        accuracy_score, threshold=0.1, label_type="authenticity"
    )
    assert pytest.approx(res_08.scores["overall"]["accuracy_score"]) == 0.6875
    assert pytest.approx(res_01.scores["overall"]["accuracy_score"]) == 0.5625
