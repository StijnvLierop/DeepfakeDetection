import torch
import torchvision
from torchvision.transforms import transforms

from deepfake_detection.models.model import Model, TrainableModel
from deepfake_detection.models.prediction import Prediction


class ResNet50(TrainableModel):
    """
    Naive detector model that uses a ResNet50 backbone pretrained on ImageNet where the
    last classification layer is replaced and finetuned on the deepfake detection task.
    """

    @property
    def trainable_model(self):
        return self.model

    def __init__(self, weights_path: str=None, device: str='cuda:0'):
        super(ResNet50, self).__init__(name='ResNet50')
        self.model = None
        self.device = device
        self.weights_path = weights_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(224) # Based on requirements of pretrained ResNet50 model
        ])

    def load_model(self):
        # Define model
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.to(self.device)

        # If weights, load weights
        if self.weights_path:
            model.load_state_dict(torch.load(self.weights_path, weights_only=True))
            self.model = model
        else:
            self.model = model

    def predict(self, instance) -> Prediction:
        # Load model if needed
        if self.weights_path:
            self.load_model()
        else:
            raise(ValueError, "No weights provided.")

        # Make prediction
        input = self.transform(instance.data)
        output = self.model(input.to(self.device).unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return Prediction(classification={'real': float(probabilities[0]), 'fake': float(probabilities[1])})

    def prepare_for_training(self):

        # Load model
        self.load_model()

        # Freeze all layers except new ones
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():  # Unfreeze new layers
            param.requires_grad = True