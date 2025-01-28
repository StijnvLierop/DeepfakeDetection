import os

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from stqdm import stqdm
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from deepfake_detection.utils.io import write_predictions_to_file, get_predictions_filename, read_predictions_from_file
from deepfake_detection.utils.parameters import DATASETS, MODELS

# Page layout
st.set_page_config(layout="wide")

# Dataset selection
st.write("Experiment parameters")
selected_dataset = st.selectbox("Dataset", DATASETS.keys())
dataset = DATASETS[selected_dataset]

# Model selection
selected_model = st.selectbox("Model", MODELS.keys())
model = MODELS[selected_model]

# Check if prediction results already exist
result_dir = os.path.join(os.getcwd(), 'results')
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
predictions_file = os.path.join(result_dir, get_predictions_filename(dataset, model.name))

# Only make predictions when necessary
if not os.path.exists(predictions_file):
    predict_button = st.button("Make Predictions")
    if predict_button:
        predictions = [model.predict(i) for i in stqdm(dataset, total=len(dataset))]
        write_predictions_to_file(result_dir, predictions, dataset, model.name)
predictions_df = read_predictions_from_file(predictions_file)

# Show predictions
real_class_label = st.selectbox('Real class label', predictions_df['ground_truth'].unique())
binary_labels = [0 if l == real_class_label else 1 for l in predictions_df['ground_truth']]
predictions = np.where(predictions_df['prediction'].to_numpy() > 0, 1, 0).flatten()

cf_matrix, class_report = st.columns(2)
with cf_matrix:
    # Create confusion matrix
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay.from_predictions(binary_labels, predictions, ax=ax)
    st.pyplot(fig)

with class_report:
    st.dataframe(pd.DataFrame(classification_report(binary_labels, predictions, output_dict=True)).T)

# Show misclassifications
fp_column, fn_column = st.columns(2)

with fp_column:
    # Get some false positives
    false_positives = predictions_df[(predictions_df['ground_truth'] == real_class_label) & (predictions_df['prediction'] > 0)].sample(5)
    # Show images
    st.write('False positives:')
    for key, row in false_positives.iterrows():
        st.image(row['file'])

with fn_column:
    # Get some false negatives
    false_negatives = predictions_df[(predictions_df['ground_truth'] != real_class_label) & (predictions_df['prediction'] <= 0)].sample(5)
    # Show images
    st.write('False negatives:')
    for key, row in false_negatives.iterrows():
        st.image(row['file'])

