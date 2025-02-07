from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from deepfake_detection.data.datasets.dataset import Dataset
from deepfake_detection.models.prediction import Prediction


def run_tsne(dataset: Dataset, predictions: Sequence[Prediction]) -> pd.DataFrame:
    """
    This function calculates a T-SNE given a model and a dataset. The model should support getting the features.

    :param dataset: The dataset to pass through the model.
    :param predictions: The predictions containing the features.
    :return: A dataframe containing the T-SNE features and associated labels for the given dataset and model.
    """
    # Get feature representations
    features = [p.embedding for p in predictions]
    labels = [i.label for i in dataset]

    # Run T-SNE on features
    features_embedded = (TSNE(n_components=2, learning_rate='auto', init='random')
                         .fit_transform(np.array(features)))

    # Transform to dataframe
    tsne_df = pd.DataFrame({'x': features_embedded[:, 0],
                            'y': features_embedded[:, 1],
                            'label': labels}
                           )

    return tsne_df