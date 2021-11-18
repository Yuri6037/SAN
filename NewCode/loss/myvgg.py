import torchvision
import torch.nn as nn

from . import CX_distance


class MyVGG(nn.Module):
    def __init__(self):
        super(MyVGG, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=True).features

    def forward(self, sr, hr):
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        loss = CX_distance.CX_loss(hr_features, sr_features)
        return loss
