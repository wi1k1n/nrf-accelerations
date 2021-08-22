import torch
from fairnr.data.data_utils import load_exr
from fairnr.criterions.flip_loss import HDRFLIPLoss

loss = HDRFLIPLoss()

dark = torch.Tensor(load_exr('data/blender/guitar_static_exr/rgb/0151.exr')).permute((2, 0, 1)).unsqueeze(0).cuda()

loss(dark, dark)

print(dark)