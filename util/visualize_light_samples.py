import argparse, sys, os, os.path as op, json, subprocess
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

PATH = 'D:\\edu\\UniBonn\\Study\\thesis\\codes\\NSVF\\'
# PATH2MODEL = 'D:\\edu\\UniBonn\\Study\\thesis\\codes\\blender\\projects\\brdf_sphere\\brdf_sphere.ply'

plotData = []


light_start = np.load(op.join(PATH, 'light_start.npy'))
light_dirs = np.load(op.join(PATH, 'light_dirs.npy'))
hits = np.load(op.join(PATH, 'hits.npy'))

# sample_xyz = ray_start + ray_dir * sampled_depth
# sample_xyz = sample_xyz[np.tile(sample_mask, sample_mask + (3,))].reshape(sample_xyz.shape)

# light_start = light_start[39, :25, :]
# light_dirs = light_dirs[39, :25, :]

light_start = light_start[:5, ...]
light_dirs = light_dirs[:5, ...]
hits = hits[:5, ...]

for i, ls in enumerate(light_start):
	cv = ls[hits[i] > 0]
	plotData.append(go.Scatter3d(x=cv[:, 0], y=cv[:, 1], z=cv[:, 2],
								name='v{}'.format(i),
								marker=dict(size=1, color="blue"),
								mode='markers')
					)
for i, d in enumerate(light_dirs):
	cd = d[hits[i] > 0]
	cd /= np.linalg.norm(cd, axis=0)
	cv = light_start[i][hits[i] > 0]
	cvt = cv + cd
	for j, cp in enumerate(cv):
		plotData.append(go.Scatter3d(
							x=[cp[0], cvt[j, 0]],
							y=[cp[1], cvt[j, 1]],
							z=[cp[2], cvt[j, 2]],
							name='pts',
							marker=dict(size=1, color="red"),
							mode='lines')
						)

fig = go.Figure(data=plotData)
print('Saving to {0}'.format(os.path.abspath('visualize_light_samples.html')))
fig.write_html('visualize_light_samples.html', auto_open=True)