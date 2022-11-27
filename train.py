import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sde import VariancePreservingSDE, PluginReverseSDE
from utils import get_grid
from utils import LogitTransform
from UNET import UNet
from utils import logging, create
from tensorboardX import SummaryWriter
import json


_folder_name_keys = ['dataset', 'real', 'debias', 'batch_size', 'lr', 'num_iterations']



# i/o
dataset='mnist'
dataroot='./datasets'
saveroot ='./saved'
expname = 'default'
print_every= 500
sample_every=500
checkpoint_every=1000
num_steps= 1000
FID_every = 1000
num_iterations=10000
# optimization
T0=1.0
vtype= 'rademacher'
batch_size=64
test_batch_size=256
lr=0.0001
# model
real=True
debias=False


folder_tag = 'sde-flow'
folder_name = '-'.join([ k for k in _folder_name_keys])
create(saveroot, folder_tag, expname, folder_name)
folder_path = os.path.join(saveroot, folder_tag, expname, folder_name)
print_ = lambda s: logging(s, folder_path)
print_(f'folder path: {folder_path}')
writer = SummaryWriter(folder_path)

input_channels = 1
input_height = 28
dimx = input_channels * input_height ** 2

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root=dataroot, train=True,
                                      download=True, transform=transform)
testset = torchvision.datasets.MNIST(root=dataroot, train=False,
                                      download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                          shuffle=True, num_workers=2)

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

cuda = torch.cuda.is_available()
if cuda:
    sde_model.cuda()

optim = torch.optim.Adam(sde_model.parameters(), lr=lr)

logit = LogitTransform(alpha=0.05)
if real:
    reverse = logit.reverse
else:
    reverse = None
num_iterations=10000
@torch.no_grad()
def evaluate(gen_sde):
    test_bpd = list()
    gen_sde.eval()
    for x_test, _ in testloader:
        if cuda:
            x_test = x_test.cuda()
        x_test = x_test * 255 / 256 + torch.rand_like(x_test) / 256
        if real:
            x_test, ldj = logit.forward_transform(x_test, 0)
            elbo_test = gen_sde.elbo_random_t_slice(x_test)
            elbo_test += ldj
        else:
            elbo_test = gen_sde.elbo_random_t_slice(x_test)
        test_bpd.extend(- (elbo_test.data.cpu().numpy() / dimx) / np.log(2) + 8)
         
    
    gen_sde.train()
    test_bpd = np.array(test_bpd)
    return test_bpd.mean(), test_bpd.std() / len(testloader.dataset.data) ** 0.5

@torch.no_grad()
def generate(gen_sde):
    print("Beginning of Generation ....")
    for j in range(1024):    
        with torch.no_grad():
                  print("Begin Generation of image ....",j)
                  mean=0
                  std=1
                  delta = gen_sde.T / 50
                  y0 = torch.randn(1, input_channels, input_height, input_height).to(gen_sde.T)
                  y0 = y0 * std + mean
                  ts = torch.linspace(0, 1, 50 + 1).to(y0) * gen_sde.T
                  ones = torch.ones(1, 1, 1, 1).to(y0)
                  for i in range(50):
                        mu = gen_sde.mu(ones * ts[i], y0)
                        sigma = gen_sde.sigma(ones * ts[i], y0)
                        y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)
                  y0 = torch.clip(y0, 0, 1)
                  y0 = y0.view(1, 1, input_channels, input_height, input_height).permute(2, 0, 3, 1, 4).contiguous().view(input_channels, 1 * input_height, 1 * input_height)
                  
                  torchvision.utils.save_image(y0, os.path.join('./generated/', f'{j}.png'))
                  print("Generation of image ....",j)
    print("End of Generation ....")
     
if os.path.exists(os.path.join(folder_path, 'checkpoint.pt')):
    gen_sde, optim, not_finished, count = torch.load(os.path.join(folder_path, 'checkpoint.pt'))
else:

    not_finished = True

    count = 0

    writer.add_scalar('T', sde_model.T.item(), count)

    writer.add_image('samples',
                     get_grid(sde_model, input_channels, input_height, n=4,
                              num_steps= num_steps, transform=reverse),
                     0)  
while not_finished:

    for x, _ in trainloader:

        if cuda:
            x = x.cuda()
        x = x * 255 / 256 + torch.rand_like(x) / 256
        if real:
            x, _ = logit.forward_transform(x, 0)

        loss = sde_model.dsm(x).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        count += 1
        print("step .....",count)
        if count == 1 or count % print_every == 0:
            writer.add_scalar('loss', loss.item(), count)
            writer.add_scalar('T', sde_model.T.item(), count)
            bpd, std_err = evaluate(sde_model)
            writer.add_scalar('bpd', bpd, count)
            print_(f'Iteration {count} \tBPD {bpd}')

        if count >= num_iterations:
            not_finished = False
            print_('Finished training')
            break

        if count % sample_every == 0:
            sde_model.eval()
            writer.add_image('samples',
                             get_grid(sde_model, input_channels, input_height, n=4,
                                      num_steps=num_steps, transform=reverse),
                             count)
            sde_model.train()
            print("sampling",count)

        if count % checkpoint_every == 0:
            torch.save([sde_model, optim, not_finished, count], os.path.join(folder_path, 'checkpoint.pt'))
            print("checkpoint",count)
torch.save(sde_model, os.path.join('/content/drive/MyDrive/Colab Notebooks/modified version/Checkpoint', 'bestmodel.pth'))