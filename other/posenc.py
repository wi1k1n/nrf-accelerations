from pysmtb.iv import iv

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

input_dir = '../data/images/'
log_dir = '../logs/implicit/'

for exp in range(10000):
    exp_dir = os.path.join(log_dir, 'ver_%05d' % exp)
    if not os.path.exists(exp_dir):
        break
else:
    raise Exception('clean up log dir ' + log_dir)
os.makedirs(exp_dir)

im = np.array(Image.open(os.path.join(input_dir, 'lena.png'))).astype(np.float32) / 255.
h, w = im.shape[:2]

N = h * w
L = 14
H = 128
C = im.shape[2]
lr_init = 1e-3
lr_final = 1e-4
num_cycles = 20
B = h
max_epoch = 1000
pred_interval = 10
ss = 8
N_ss = h * w * ss * ss
B_ss = ss * h

def pos_encode(x, L):
    if L == 0:
        return x
    else:
        return torch.stack([e for es in [[torch.cos(l * x), torch.sin(l * x)] for l in range(1, L + 1)] for e in es], axis=-1)

# ys, xs = torch.meshgrid(torch.linspace(0, h-1, h), torch.linspace(0, w-1, w))
ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
xys = torch.stack((xs, ys), axis=2).reshape(-1, 2)
xys_enc = pos_encode(xys, L)
xys_enc = xys_enc.reshape((h, w, 2, -1))

ys_super, xs_super = torch.meshgrid(torch.linspace(-1, 1, h * ss), torch.linspace(-1, 1, w * ss))
xys_super = torch.stack((xs_super, ys_super), axis=2).reshape(-1, 2)
xys_super_enc = pos_encode(xys_super, L)
xys_super_enc = xys_super_enc.reshape((h * ss, w * ss, 2, -1))

inputs = xys_enc.reshape(N, -1)
labels = torch.tensor(im.reshape(N, C))

inputs_ss = xys_super_enc.reshape(N_ss, -1)

# iv(xys_enc.detach()[:, :, :, :].permute((3, 2, 1, 0)))

class Sin(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Model(torch.nn.Module):
    def __init__(self, L=14, H=128):
        super().__init__()

        nc_in = 4 * L
        if L == 0:
            nc_in = 2
        layers = [torch.nn.Linear(nc_in, H),
                  #torch.nn.LeakyReLU(),
                  torch.nn.ReLU(),
                  #Sin(),
                  torch.nn.Linear(H, H),
                  #torch.nn.LeakyReLU(),
                  torch.nn.ReLU(),
                  #Sin(),
                  torch.nn.Linear(H, H),
                  #torch.nn.LeakyReLU(),
                  torch.nn.ReLU(),
                  #Sin(),
                  torch.nn.Linear(H, C),
                  #torch.nn.LeakyReLU(),
                  torch.nn.ReLU(),
                  ]

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        outputs = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            outputs.append(x)
        return x, outputs

model = Model(L, H)

def cycle_decay(max_steps=400, lower=1e-5, upper=1e-3, cycles=50):
    from math import exp, sin, pi
    decay_steps = max_steps / 1.5
    scale = lower / upper
    cycles = min(cycles, max_steps / 4)
    return lambda step: scale + (1 - scale) * exp(-2.3 * step ** 2 / (decay_steps ** 2)) \
                        * (sin(2 * pi * cycles * step / max_steps) + 1) / 2
lr_step = cycle_decay(max_steps=max_epoch,
                      lower=lr_final,
                      upper=lr_init,
                      cycles=num_cycles)
optim = torch.optim.Adam(params=model.parameters(), lr=lr_init)
sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_step)# , last_epoch=max_epoch)
criterion = torch.nn.L1Loss()

writer = SummaryWriter(log_dir=exp_dir, comment='visplicit experiment')
predictions = []

progress = tqdm(None, desc='training', total=max_epoch, ncols=100, unit='epoch')

global_step = 0
for ep in tqdm(range(max_epoch), 'epochs'):
    perm = torch.randperm(N)
    inps = inputs[perm, :]
    lbls = labels[perm, :]
    for it in range(0, N, B):
        pred, _ = model(inps[it: it+B, :])
        loss = criterion(pred, lbls[it:it+B, :])

        optim.zero_grad()
        loss.backward()
        optim.step()

        writer.add_scalar('loss/train', loss.item(), global_step=global_step)
        global_step += 1
    sched.step()
    writer.add_scalar('lr', sched.get_lr()[-1], global_step=global_step)

    progress.set_description()
    progress.update(1)
    progress.set_description('loss: %f' % loss.item())
    if ep % pred_interval == 0:
        prediction, _ = model(inputs)
        predictions.append(prediction.detach().numpy().reshape((h, w, C)))
writer.close()

with torch.no_grad():
    prediction_ss = torch.zeros(N_ss, C)
    intermediates = {}
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], torch.nn.Linear):
            intermediates[i] = torch.zeros(N_ss, model.layers[i].out_features)
    for it in tqdm(range(0, N_ss, B_ss), 'predicting'):
        pred, intermed = model(inputs_ss[it: it+B_ss, :])
        prediction_ss[it: it+B_ss, :] = pred
        for i in range(len(model.layers)):
            if isinstance(model.layers[i], torch.nn.Linear):
                intermediates[i][it: it+B_ss, :] = intermed[i]
prediction_ss = prediction_ss.detach().cpu().numpy().reshape((h * ss, w * ss, C))
for i in intermediates.keys():
    intermediates[i] = intermediates[i].reshape(h * ss, w * ss, 1, -1).cpu().numpy()
iv(im, prediction_ss)
iv(intermediates[4][::4, ::4, :, :])
