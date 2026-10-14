"""DenseNet121 (ImageNet pretrained) + dropout 0.5 + 6-class classifier."""
import torch.nn as nn
from torchvision import models

import config as C


def build_model(num_classes=C.NUM_CLASSES, dropout=C.DROPOUT, pretrained=True, memory_efficient=False):
    """memory_efficient=True: dense block에 gradient checkpointing 적용 (결과 동일, 메모리↓ 속도↓)."""
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.densenet121(weights=weights, memory_efficient=memory_efficient)
    in_features = m.classifier.in_features  # 1024
    m.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return m
