"""Frozen VGG16 feature L1, separate from the SqueezeNet-LPIPS evaluator."""
import torch
from torch import nn
from torchvision.models import vgg16,VGG16_Weights

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        backbone=vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features=backbone.features[:16].eval()
        for p in self.features.parameters():p.requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def encode(self,image):
        # Inputs are composed images in [-1,1]. No training-time clipping.
        h=((image.float()+1)/2-self.mean)/self.std;outputs=[]
        for i,layer in enumerate(self.features):
            h=layer(h)
            if i in [3,8,15]:outputs.append(h)
        return outputs
    def per_image(self,prediction,truth,mask):
        composed=truth*(1-mask)+prediction*mask
        pred=self.encode(composed)
        with torch.no_grad():target=self.encode(truth)
        return torch.stack([(p.float()-t.float()).abs().flatten(1).mean(1)for p,t in zip(pred,target)]).mean(0)

def terms(net,perceptual,x,m,coefficient):
    from pilot import autocast
    y=x*(1-m);t=torch.zeros(len(x),device=x.device)
    with autocast():
        pred=net(y,y,m,t).float()
        pixel=((pred-x).square()*m).sum((1,2,3))/(3*m.sum((1,2,3)))
        feature=perceptual.per_image(pred,x,m)
    return {'mse':pixel,'feature_l1':feature,'weighted_feature':coefficient*feature,'total':pixel+coefficient*feature}
