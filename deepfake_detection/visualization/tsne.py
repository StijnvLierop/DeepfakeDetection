from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import streamlit as st

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.models.prediction import Prediction
from deepfake_detection.utils.parameters import DATASETS


hash_funcs = {d: lambda x: x.__hash__() for d in DATASETS.values()}
hash_funcs[Prediction] = lambda x: x.__hash__()

@st.cache_data(hash_funcs=hash_funcs)
def run_tsne(dataset: Dataset, predictions: Sequence[Prediction]) -> pd.DataFrame:
    """
    This function calculates a T-SNE given a model and a dataset. The model should support getting the features.

    :param dataset: The dataset to pass through the model.
    :param predictions: The predictions containing the features.
    :return: A dataframe containing the T-SNE features and associated labels for the given dataset and model.
    """
    # Get feature representations
    features = [p.embedding for p in predictions]
    labels = [i.class_label for i in dataset]
    filepaths = [i.path for i in dataset]

    # Run T-SNE on features
    features_embedded = (TSNE(n_components=2, learning_rate='auto', init='random')
                         .fit_transform(np.array(features)))

    # Transform to dataframe
    tsne_df = pd.DataFrame({'filepath': filepaths,
                            'x': features_embedded[:, 0],
                            'y': features_embedded[:, 1],
                            'label': labels}
                           )

    return tsne_df