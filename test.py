import torch.nn as nn
import torch
dummy_data=torch.randn(1, 16, 12, 12)
up=nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1,output_padding=1)
output = up(dummy_data)
print(output.shape)
out_channels=[2**(i+6) for i in range(5)]
print(out_channels)