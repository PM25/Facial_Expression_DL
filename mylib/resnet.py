import torch
import torchvision
from torch import nn


class ResNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        fc_in_size = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_in_size, n_classes)


    def forward(self, x):
        # No Need Softmax because it's included in nn.CrossEntropyLoss()
        output = self.model(x)
        return output