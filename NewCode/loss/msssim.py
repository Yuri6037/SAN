import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.nn as nn

class MSSSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, sr, hr):
        if torch.max(sr) > 1:
            sr = torch.div(sr, 255.0) # normalize tensor if not already
        hr = hr.to(self.device, dtype=torch.float32)
        hr = torch.div(hr, 255.0) # normalize tensor
        loss = 1 - ssim(sr, hr, data_range=1, size_average=True)
        return loss
