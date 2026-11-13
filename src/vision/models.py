"""Common model/checkpoint contract. DINOv2 backbone is always frozen."""
from __future__ import annotations
import torch
from torch import nn
from .data import CLASSES

DINO_REVISION='ed25f3a31f01632728cabb09d1542f84ab7b0056'


class FrozenDINO(nn.Module):
    def __init__(self,backbone,dropout=.5):
        super().__init__(); self.backbone=backbone
        self.backbone.requires_grad_(False); self.backbone.eval()
        self.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(backbone.config.hidden_size*2,len(CLASSES)))
    def train(self,mode=True):
        super().train(mode); self.backbone.eval();return self
    def features(self,x):
        h=self.backbone(pixel_values=x).last_hidden_state
        return torch.cat((h[:,0],h[:,1:].mean(dim=1)),dim=1)
    def forward(self,x):return self.classifier(self.features(x))


def build_model(arch,pretrained=False,dino_config=None,dropout=.5):
    from torchvision import models
    if arch=='dinov2_linear':
        from transformers import Dinov2Model,Dinov2Config
        if pretrained:
            backbone=Dinov2Model.from_pretrained('facebook/dinov2-small',revision=DINO_REVISION,use_safetensors=True,local_files_only=True)
        else:
            if dino_config is None:raise ValueError('checkpoint DINO configuration required')
            backbone=Dinov2Model(Dinov2Config(**dino_config))
        return FrozenDINO(backbone,dropout)
    if arch=='densenet121':
        model=models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(1024,len(CLASSES)))
    elif arch=='efficientnet_b0':
        model=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(1280,len(CLASSES)))
    else:raise ValueError('unknown architecture: '+arch)
    return model


def checkpoint_config(model,arch,size):
    return {'architecture':arch,'image_size':size,'class_names':list(CLASSES),'dropout':float(model.classifier[0].p),
            'dino_config':model.backbone.config.to_dict() if isinstance(model,FrozenDINO) else None,
            'dino_revision':DINO_REVISION if isinstance(model,FrozenDINO) else None,
            'preprocessing':'RGB resize square; ImageNet normalization'}


def load_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    cfg=checkpoint['config']
    if tuple(cfg['class_names'])!=CLASSES:raise ValueError('checkpoint class order mismatch')
    model=build_model(cfg['architecture'],dino_config=cfg.get('dino_config'),dropout=cfg['dropout'])
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.to(device).eval(),checkpoint
