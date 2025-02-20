import os

import streamlit as st
from stqdm import stqdm
import plotly.express as px

from deepfake_detection.utils.configuration import parse_dataset_config, parse_model_config
from deepfake_detection.utils.io import write_predictions_to_file, get_predictions_filename, read_predictions_from_file
from deepfake_detection.visualization.tsne import run_tsne


# Load datasets and models
datasets = parse_dataset_config('dataset_config.yaml')
models = parse_model_config('model_config.yaml')

# Page layout
st.set_page_config(layout="wide")

# Dataset selection
st.write("Experiment parameters")
selected_dataset = st.selectbox("Dataset", datasets.keys())
dataset = datasets[selected_dataset]

# Model selection
selected_model = st.selectbox("Model", models.keys())
model = models[selected_model]

# Check if prediction results already exist
result_dir = os.path.join(os.getcwd(), 'results')
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
predictions_file = os.path.join(result_dir, get_predictions_filename(dataset.name, model.name))

# Only make predictions when necessary
predictions = None
if not os.path.exists(predictions_file):
    predict_button = st.button("Make Predictions")
    if predict_button:
        predictions = [model.predict(i) for i in stqdm(dataset, total=len(dataset))]
        write_predictions_to_file(result_dir, predictions, dataset, model.name)
else:
    predictions = read_predictions_from_file(predictions_file)

if predictions is not None:

    plot_col, img_col = st.columns(2)

    with plot_col:
        # Show T-SNE
        st.header("T-SNE")
        tsne_df = run_tsne(dataset, predictions)
        tsne_fig = px.scatter(data_frame=tsne_df, x='x', y='y', color='label', height=800, width=1000)
        events = st.plotly_chart(tsne_fig, on_select="rerun")

    with img_col:
        # Show images of clicked datapoints
        paths = []
        if len(events['selection']['points']) > 0:
            for item in events['selection']['points']:
                path = tsne_df.loc[(tsne_df['x'] == item['x']) & (tsne_df['y'] == item['y']), 'filepath'].to_list()[0]
                paths.append(path)

        # for path in paths:
        for path in paths:
            st.image(path, width=1000)