# Alzheimer_CNN/src/model.py
import torch
import torch.nn as nn
from torchvision import models

class AlzheimerCNN(nn.Module):
    def __init__(self):
        super(AlzheimerCNN, self).__init__()
        self.network = models.resnet18(weights=None)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.network(x)

def train_or_load_model(device):
    model = AlzheimerCNN().to(device)
    model.eval()
    return model
