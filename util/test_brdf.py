from fairnr.modules.brdf import Microfacet
import torch
import numpy as np

brdf = Microfacet()

N = 4
L = 1

np.random.seed(125)

pts2l = torch.Tensor(np.random.random((N, L, 3)))
pts2c = torch.Tensor(np.random.random((N, 3)))
normal = torch.Tensor(np.random.random((N, 3)))
albedo = torch.Tensor(np.random.random((N, 3)))
rough = torch.Tensor(np.random.random((N, 1)))

res = brdf(pts2l, pts2c, normal, albedo, rough)

print(res)