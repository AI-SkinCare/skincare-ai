"""Class-specific Grad-CAM. A sensitivity map is not a lesion annotation."""
from __future__ import annotations
import base64
import io
import math
import torch
from torch.nn import functional as F


def target_layer(model,architecture):
    if architecture=='densenet121':return model.features.norm5,'features.norm5'
    if architecture=='efficientnet_b0':return model.features[-1],'features[-1]'
    if architecture=='dinov2_linear':return model.backbone.encoder.layer[-1].norm1,'backbone.encoder.layer[-1].norm1'
    raise ValueError('unsupported Grad-CAM architecture')


def grad_cam(model,x,architecture,class_index=None):
    layer,name=target_layer(model,architecture); activations=[]
    # No parameter unfreezing: input gradients make the frozen encoder traceable.
    def capture(_,inputs,output):activations.append(output)
    hook=layer.register_forward_hook(capture)
    try:
        model.eval()
        with torch.enable_grad():
            x=x.detach().requires_grad_(True)
            logits=model(x)
            index=int(logits.argmax(1)[0]) if class_index is None else int(class_index)
            if not 0<=index<logits.shape[1]:raise ValueError('invalid explanation target class')
            a=activations[0]
            g=torch.autograd.grad(logits[0,index],a,retain_graph=False)[0]
            if a.ndim==3:
                side=math.isqrt(a.shape[1]-1)
                if side*side!=a.shape[1]-1:raise ValueError('DINO patch grid is not square')
                a=a[:,1:].transpose(1,2).reshape(1,-1,side,side)
                g=g[:,1:].transpose(1,2).reshape(1,-1,side,side)
            cam=(g.mean((2,3),keepdim=True)*a).sum(1,keepdim=True).relu().detach()
            cam=F.interpolate(cam,size=(256,256),mode='bilinear',align_corners=False)[0,0]
            span=float(cam.max()-cam.min())
            available=bool(torch.isfinite(cam).all()) and span>1e-12
            cam=(cam-cam.min())/span if available else torch.zeros_like(cam)
        return cam.cpu(),logits.detach().float().softmax(1)[0].cpu().tolist(),index,name,available
    finally:hook.remove()


def explain_bytes(model,x,image_bytes,architecture):
    from PIL import Image
    cam,probabilities,index,layer,available=grad_cam(model,x,architecture)
    gray=(cam.numpy()*255).round().astype('uint8')
    heat=Image.fromarray(gray,mode='L')
    def encode(img):
        b=io.BytesIO();img.save(b,format='PNG');return base64.b64encode(b.getvalue()).decode('ascii')
    return {'method':'grad_cam','target_class_index':index,'target_layer':layer,'probabilities':probabilities,
            'available':available,'heatmap_png_base64':encode(heat) if available else None,
            'interpretation':'모델 분류 점수에 영향을 준 영역이며 병변의 정답 위치·중증도·진단 근거를 확정하지 않습니다.'}
