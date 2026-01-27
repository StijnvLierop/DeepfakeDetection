import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from deepfake_detection.evaluation.evaluator import Evaluator
from deepfake_detection.data.instance import ImageInstance
from deepfake_detection.data.annotation import Annotation
from deepfake_detection.models.prediction import Prediction
from PIL import Image

def test_evaluator_run_basic():
    # Setup
    img = Image.new('RGB', (10, 10))
    instances = [
        ImageInstance(img, annotation=Annotation({"label": "real"})),
        ImageInstance(img, annotation=Annotation({"label": "fake"})),
    ]
    predictions = [
        Prediction(classification={"real": 0.9, "fake": 0.1}),
        Prediction(classification={"real": 0.2, "fake": 0.8}),
    ]
    evaluator = Evaluator(instances, predictions)
    
    # Run
    result = evaluator.run(metrics=accuracy_score, label_type="label")
    
    # Verify
    assert "all" in result.scores
    # accuracy_score.__name__ is 'accuracy_score'
    # For "all", it calculates macro average of "real" and "fake"
    # For "real": y_true=[1, 0], y_pred=[1, 0] (threshold 0.5) -> accuracy 1.0
    # For "fake": y_true=[0, 1], y_pred=[0, 1] (threshold 0.5) -> accuracy 1.0
    # Macro average: 1.0
    assert 1.0 == result.scores["all"]["accuracy_score"]

def test_evaluator_run_multiple_metrics():
    img = Image.new('RGB', (10, 10))
    instances = [
        ImageInstance(img, annotation=Annotation({"label": "real"})),
        ImageInstance(img, annotation=Annotation({"label": "fake"})),
    ]
    predictions = [
        Prediction(classification={"real": 0.9, "fake": 0.1}),
        Prediction(classification={"real": 0.2, "fake": 0.8}),
    ]
    evaluator = Evaluator(instances, predictions)
    
    result = evaluator.run(metrics=[accuracy_score, roc_auc_score], label_type="label")
    
    assert "accuracy_score" in result.scores["all"]
    assert "roc_auc_score" in result.scores["all"]
    assert 1.0 == result.scores["all"]["accuracy_score"]
    assert 1.0 == result.scores["all"]["roc_auc_score"]

def test_evaluator_run_target_class():
    img = Image.new('RGB', (10, 10))
    instances = [
        ImageInstance(img, annotation=Annotation({"label": "real"})),
        ImageInstance(img, annotation=Annotation({"label": "fake"})),
    ]
    predictions = [
        Prediction(classification={"real": 0.9, "fake": 0.1}),
        Prediction(classification={"real": 0.6, "fake": 0.4}),
    ]
    evaluator = Evaluator(instances, predictions)
    
    # target_class="real"
    # real: y_true=[1, 0], y_pred=[1, 1] (at 0.5 threshold) -> accuracy 0.5
    result = evaluator.run(metrics=accuracy_score, label_type="label", target_class="real")
    
    assert 0.5 == result.scores["all"]["accuracy_score"]

def test_evaluator_run_group_by():
    img = Image.new('RGB', (10, 10))
    instances = [
        ImageInstance(img, annotation=Annotation({"label": "real"}), meta={"source": "cam1"}),
        ImageInstance(img, annotation=Annotation({"label": "fake"}), meta={"source": "cam2"}),
    ]
    predictions = [
        Prediction(classification={"real": 0.9, "fake": 0.1}),
        Prediction(classification={"real": 0.2, "fake": 0.8}),
    ]
    evaluator = Evaluator(instances, predictions)
    
    result = evaluator.run(metrics=accuracy_score, label_type="label", group_by="source")
    
    assert "cam1" in result.scores
    assert "cam2" in result.scores
    # cam1: only "real" class. 
    # _calculate_macro_average will be called for "cam1" slice.
    # labels for "cam1" slice: get_labels([cam1_inst], [cam1_pred], "label") -> ["fake", "real"]
    # fake: y_true=[0], y_pred=[0] -> accuracy 1.0
    # real: y_true=[1], y_pred=[1] -> accuracy 1.0
    # macro: 1.0
    assert 1.0 == result.scores["cam1"]["accuracy_score"]
    assert 1.0 == result.scores["cam2"]["accuracy_score"]

def test_evaluator_run_negative_class_label():
    img = Image.new('RGB', (10, 10))
    instances = [
        ImageInstance(img, annotation=Annotation({"label": "real"}), meta={"source": "cam1"}),
        ImageInstance(img, annotation=Annotation({"label": "fake"}), meta={"source": "cam2"}),
        ImageInstance(img, annotation=Annotation({"label": "real"}), meta={"source": "cam2"}),
    ]
    predictions = [
        Prediction(classification={"real": 0.9, "fake": 0.1}),
        Prediction(classification={"real": 0.1, "fake": 0.9}),
        Prediction(classification={"real": 0.8, "fake": 0.2}),
    ]
    evaluator = Evaluator(instances, predictions)
    
    # cam1 only has "real". If we specify negative_class_label="fake", it should NOT be added to cam1 if it's not in that slice?
    # Wait, the code says:
    # if len(unique_classes) < 2 and negative_class_label and slice_name != negative_class_label:
    #    slice_inst.extend([self.instances[i] for i in neg_indices])
    #    slice_pred.extend([self.predictions[i] for i in neg_indices])
    
    # In our case, for "cam1", unique_classes = ["real"]. 
    # neg_indices for negative_class_label="fake" will be [1].
    # So cam1 slice will become [instances[0], instances[1]].
    
    result = evaluator.run(metrics=roc_auc_score, label_type="label", group_by="source", negative_class_label="fake")
    
    assert "cam1" in result.scores
    # cam1 slice now has instances[0] (real) and instances[1] (fake).
    # predictions[0] (real:0.9), predictions[1] (fake:0.9 -> real:0.1)
    # real: y_true=[1, 0], y_prob=[0.9, 0.1] -> AUC 1.0
    # fake: y_true=[0, 1], y_prob=[0.1, 0.9] -> AUC 1.0
    assert 1.0 == result.scores["cam1"]["roc_auc_score"]

def test_evaluator_run_invalid_group_by():
    img = Image.new('RGB', (10, 10))
    instances = [ImageInstance(img, annotation=Annotation({"label": "real"}))]
    predictions = [Prediction(classification={"real": 0.9, "fake": 0.1})]
    evaluator = Evaluator(instances, predictions)
    
    with pytest.raises(ValueError, match="Key 'invalid_key' not found."):
        evaluator.run(metrics=accuracy_score, label_type="label", group_by="invalid_key")
