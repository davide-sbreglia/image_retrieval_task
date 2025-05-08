import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class L2Norm(nn.Module):
    def forward(self, x):
        # normalize along the feature-dimension
        return F.normalize(x, p=2, dim=1)

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
        L2Norm()  # L2-normalize embeddings
    )