import torch
from torch import nn
from torchvision import models

def build_model(num_classes):
    # 1) load pretrained EfficientNet-B0
    eff = models.efficientnet_b0(pretrained=True)
    # 2) freeze all backbone weights
    for param in eff.features.parameters():
        param.requires_grad = False

    # 3) replace classifier head
    in_features = eff.classifier[1].in_features
    eff.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return eff

def extract_embedding_model(model):
    # strip off the classifier head
    return nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.functional.normalize  # L2-normalize embeddings
    )