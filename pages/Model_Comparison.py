import os

import pandas as pd
import streamlit as st

from deepfake_detection.evaluation.metrics import roc_auc, accuracy
from deepfake_detection.utils.configuration import parse_dataset_config
from deepfake_detection.utils.io import read_predictions_from_file

datasets = parse_dataset_config('dataset_config.yaml')

results = []

for predictions_file in os.listdir("./results"):
    _, dataset_name, model_name = predictions_file.split('.')[0].split("_")
    predictions = read_predictions_from_file(os.path.join("./results", predictions_file))
    dataset = datasets[dataset_name]

    acc = accuracy(dataset, predictions, label_mapping=dataset.label_mapping)
    auc = roc_auc(dataset, predictions, label_mapping=dataset.label_mapping)

    results.append([dataset_name, model_name, acc, auc])

results_df = pd.DataFrame(results, columns=['dataset_name', 'model_name', 'accuracy', 'AUC'])
st.write(results_df)