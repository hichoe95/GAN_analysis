import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from torchvision import models
import numpy as np
from math import log10

class VGG16_perceptual(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16_perceptual, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 4):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu1_2 = h
        h = self.slice3(h)
        h_relu3_2 = h
        h = self.slice4(h)
        h_relu4_2 = h
        return h_relu1_1, h_relu1_2, h_relu3_2, h_relu4_2
    
    
def loss_fn(gen_img, real, criterion, upsample, perceptual):
    gen_img_p = upsample(gen_img)
    real = real.permute(2,0,1).unsqueeze(0)
    real_p = real.clone()
    real_p = real_p
    real_p = upsample(real_p)

    gen0, gen1, gen2, gen3 = perceptual(gen_img_p)
    r0, r1, r2, r3 = perceptual(real_p)

    mse = criterion(gen_img, real)

    per_loss = criterion(gen0, r0)
    per_loss += criterion(gen1, r1)
    per_loss += criterion(gen2, r2)
    per_loss += criterion(gen3, r3)

    return mse, per_loss



