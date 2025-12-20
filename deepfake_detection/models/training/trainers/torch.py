import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detection.models.model import TrainableMixin
from deepfake_detection.data.dataset import Dataset, MapStyleDatasetMixin
from deepfake_detection.data.datasets.torch import TorchDataset, TorchIterableDataset
from deepfake_detection.models.training.trainer import Trainer, TrainParams


class PytorchTrainer(Trainer):
    """
    Trainer class to train Pytorch models that support the TrainableMixin extension.
    """
    def __init__(self, model: TrainableMixin, train_params: TrainParams):
        super().__init__(model, train_params)

        # Set optimizer
        if train_params.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.get_model_parameters(), lr=train_params.lr)
        else:
            raise ValueError("Unknown optimizer: {}".format(train_params.optimizer))

        # Set criterion
        if train_params.criterion == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Unknown criterion: {}".format(train_params))

        # History
        self.history = {'train_loss': [], 'val_loss': []}

    def train(self, train_dataset: Dataset):
        """
        This function initiates the training process using a given dataset
        with the currently set class parameters.

        :param train_dataset: The training dataset.
        """

        # Get torch dataset
        if isinstance(train_dataset, MapStyleDatasetMixin):
            torch_dataset = TorchDataset(train_dataset)
        else:
            torch_dataset = TorchIterableDataset(train_dataset)
            self.train_params.shuffle = False

        # Create data loader from dataset
        loader = DataLoader(torch_dataset,
                            batch_size=self.train_params.batch_size,
                            shuffle=self.train_params.shuffle,
                            num_workers=self.train_params.num_workers)

        # Turn off eval model for training
        self.model.train()

        print("Start training model.")

        # Loop over epochs
        for epoch in range(self.train_params.epochs):

            print("Epoch {}/{}".format(epoch + 1, self.train_params.epochs))

            epoch_loss = 0.0

            # Loop over batches
            for inputs, labels in tqdm(loader, desc='Iterating over batches...'):

                # Move to device and cast to correct format
                inputs = inputs.to(self.train_params.device).float()
                labels = labels.to(self.train_params.device).float().view(-1, 1)

                # Reset gradients
                self.optimizer.zero_grad()

                # Run a forward pass
                predictions = self.model.forward_pass(inputs)

                # Calculate loss
                loss = self.criterion(predictions, labels)
                epoch_loss += loss.item()

                # Update weights
                loss.backward()
                self.optimizer.step()

            # Add metrics
            self.history['train_loss'].append(epoch_loss)

            print(f"Epoch {epoch} finished. Train loss: {epoch_loss}")

        print(f"Finished training for {self.train_params.epochs} epochs.")

    def save_checkpoint(self, path: str):
        # Save model
        self.model.save_weights(path)
