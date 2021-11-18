import torchvision
import torch.nn as nn
import torch

from . import CX_distance


class MyVGG(nn.Module):
    def __init__(self):
        super(MyVGG, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=True).features
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, sr, hr):
        if torch.max(sr) > 1:
            sr = torch.div(sr, 255.0) # normalize tensor if not already
        hr = hr.to(self.device, dtype=torch.float32)
        hr = torch.div(hr, 255.0) # normalize tensor
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        loss = CX_distance.CX_loss(hr_features, sr_features)
        return loss
