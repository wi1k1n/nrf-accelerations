import numpy as np
import os, os.path as op
import torch

def save(v, name, folder='/tmp/remdeb'):
	savePath = op.join(folder, name + '.rmdb')
	if type(v) == np.ndarray:
		print('ndarray -> ' + savePath)
		np.save(savePath, v)
	elif type(v) == torch.Tensor:
		print('tensor -> ' + savePath)
		torch.save(v, savePath)