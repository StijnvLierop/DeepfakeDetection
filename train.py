import logging
import os
from typing import Mapping, Any

import confidence
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from deepfake_detection.data.datasets.train_dataset import TrainDataset
from deepfake_detection.utils.configuration import parse_model_config, load_dataset


def train_model(config: Mapping[str, Any]) -> None:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu")

    # Load a model
    models = parse_model_config('model_config.yaml')
    model = models[config['model_name']]
    model.prepare_for_training()

    # Initialize dataset
    dataset = load_dataset(config['train_data'][0])

    # Initialize train dataset
    train_dataset = TrainDataset(base_dataset=dataset, transform=model.transform)

    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    # Setup loss function
    criterion = nn.CrossEntropyLoss()

    # Setup Optimizer
    optimizer = optim.Adam(model.model.fc.parameters(), lr=config['learning_rate'])  # Only train new layers

    # Training loop
    logging.info("Training...")
    for epoch in range(config['n_epochs']):
        for inputs, labels in train_loader:

            # Transfer data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Optimize
            optimizer.zero_grad()
            outputs = model.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        logging.info(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

    # Save model weights
    if not os.path.exists(config['weights_dir']):
        os.mkdir(config['weights_dir'])
    weights_path = os.path.join(config['weights_dir'], f"{config['model_name']}_{config['train_data'][0]['name']}.pth")
    torch.save(model.model.state_dict(), weights_path)
    logging.info(f"Weights saved in {weights_path}.")


if __name__ == '__main__':
    # Parse arguments
    config = confidence.loadf('train.yaml')['train_config']

    # Run training
    train_model(config)
