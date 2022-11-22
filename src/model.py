import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet34_Weights


def build_model(pretrained=True, fine_tune=True, num_classes=3):
    if pretrained:
        print(f'[INFO] Loading model with pretrained weights..')
        model = models.resnet34(weights=ResNet34_Weights)
    else:
        print(f'[INFO] Loading model without pretrained weights..')
        model = models.resnet34()
    if fine_tune:
        print(f'[INFO] Fine-tuning all layers..')
        for params in model.parameters():
            params.requires_grad = True
    else:
        print(f'[INFO] Freezing hidden layers..')
        for params in model.parameters():
            params.requires_grad = False
    model.fc = nn.Linear(in_features=512, out_features=num_classes)

    return model
