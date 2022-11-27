import torch 
import torchvision
import numpy as np
import os
from tqdm import tqdm, trange



from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from sde import VariancePreservingSDE, PluginReverseSDE
from UNET import UNet

input_channels = 1
input_height = 28
dimx = input_channels * input_height ** 2
real=True
debias=False
vtype= 'rademacher'
T0=1.0
model = UNet(
input_channels=input_channels,#channels
input_height=input_height, #image_size
ch=32,#dim
ch_mult=(1, 2, 2),
num_res_blocks=2,
attn_resolutions=(16,),
resamp_with_conv=True,
)
T = torch.nn.Parameter(torch.FloatTensor([T0]), requires_grad=False)

inf_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T)
sde_model = PluginReverseSDE(inf_sde, model, T, vtype=vtype, debias=debias)
sde_model=torch.load(os.path.join('./Checkpoint', 'bestmodel.pth'))
print('Model loaded.')
print("Beginning of Generation ....")
for j in range(1024):    
        with torch.no_grad():
                  print("Begin Generation of image ....",j)
                  mean=0
                  std=1
                  delta = sde_model.T / 50
                  y0 = torch.randn(1, input_channels, input_height, input_height).to(sde_model.T)
                  y0 = y0 * std + mean
                  ts = torch.linspace(0, 1, 50 + 1).to(y0) * sde_model.T
                  ones = torch.ones(1, 1, 1, 1).to(y0)
                  for i in range(50):
                        mu = sde_model.mu(ones * ts[i], y0)
                        sigma = sde_model.sigma(ones * ts[i], y0)
                        y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)
                  y0 = torch.clip(y0, 0, 1)
                  y0 = y0.view(1, 1, input_channels, input_height, input_height).permute(2, 0, 3, 1, 4).contiguous().view(input_channels, 1 * input_height, 1 * input_height)
                  
                  torchvision.utils.save_image(y0, os.path.join('./samples', f'{j}.png'))
                  print("Generation of image ....",j)
print("End of Generation ....")
     
