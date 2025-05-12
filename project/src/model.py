import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

def build_model(num_classes, model_name='efficientnet_b0'):
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def extract_embedding_model(model, model_name='efficientnet_b0'):
    if model_name == 'efficientnet_b0':
        return nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            L2Norm()
        )
    elif model_name == 'resnet50':
        return nn.Sequential(
            nn.Sequential(*list(model.children())[:-1]),
            nn.Flatten(),
            L2Norm()
        )
    elif model_name == 'vit_b_16':
        model.heads = nn.Identity()
        return nn.Sequential(
            model,
            L2Norm()
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")