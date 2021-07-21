import argparse, sys, os, os.path as op, json, subprocess
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

PATH = 'D:\\edu\\UniBonn\\Study\\thesis\\codes\\NSVF\\'
PATH2MODEL = 'D:\\edu\\UniBonn\\Study\\thesis\\codes\\blender\\projects\\brdf_sphere\\brdf_sphere.ply'

plotData = []

# reading stl/obj/ply model
mesh = None  # (vertices, I, J, K)
# looking for model file and convert it from stl, if needed
path_mesh = op.abspath(PATH2MODEL)
if op.isfile(path_mesh):
	mesh = o3d.io.read_triangle_mesh(path_mesh)
if mesh:
	vertices = np.asarray(mesh.vertices).T
	triangles = np.asarray(mesh.triangles).T
	mesh3D = go.Mesh3d(x=vertices[0], y=vertices[1], z=vertices[2], i=triangles[0], j=triangles[1], k=triangles[2],
		flatshading=True,
		colorscale=[[0, 'gold'],
					[0.5, 'mediumturquoise'],
					[1, 'magenta']],
		# Intensity of each vertex, which will be interpolated and color-coded
		intensity = vertices[0], # np.linspace(0, 1, 12, endpoint=True),
		intensitymode='cell',
		name='model',
		showscale=False)
	plotData.append(mesh3D)
else:
	print('Failed to open model file. Only PLY, OBJ and STL files are supported. Continuing without model.')



norms = np.load(op.join(PATH, 'normals.npy'))
ray_start = np.load(op.join(PATH, 'ray_start.npy'))
ray_dir = np.load(op.join(PATH, 'ray_dir.npy'))
sampled_depth = np.load(op.join(PATH, 'sampled_depth.npy'))
sample_mask = np.load(op.join(PATH, 'sample_mask.npy'))
probs = np.load(op.join(PATH, 'probs.npy'))

ray_start = np.tile(ray_start[:, None, :], (1, sampled_depth.shape[-1], 1))
ray_dir = np.tile(ray_dir[:, None, :], (1, sampled_depth.shape[-1], 1))
sampled_depth = np.tile(sampled_depth[..., None], (1, 1, 3))

sample_xyz = ray_start + ray_dir * sampled_depth
# sample_xyz = sample_xyz[np.tile(sample_mask, sample_mask + (3,))].reshape(sample_xyz.shape)

idcs = np.arange(0, 80)  # []  # [49]
if len(idcs):
	idcs = np.array(idcs)
	xs, ys, zs = sample_xyz[idcs, :, 0][sample_mask[idcs, :]], sample_xyz[idcs, :, 1][sample_mask[idcs, :]], sample_xyz[idcs, :, 2][sample_mask[idcs, :]]
	norms = norms[idcs, :, :][sample_mask[idcs, :]].reshape(-1, 3)
	probs = probs[idcs, :][sample_mask[idcs, :]]
else:
	xs, ys, zs = sample_xyz[..., 0][sample_mask], sample_xyz[..., 1][sample_mask], sample_xyz[..., 2][sample_mask]
	norms = norms[sample_mask].reshape(-1, 3)
	probs = probs[sample_mask]

plotData.append(go.Scatter3d(x=xs, y=ys, z=zs,
							name='pts',
							marker=dict(size=1, color="blue"),
							mode='markers')
				)
for i, n in enumerate(norms):
	# print(n)
	p = probs[i]
	if p < 1e-6:
		continue
	plotData.append(go.Scatter3d(
						x=[xs[i], xs[i] + n[0] * p],
						y=[ys[i], ys[i] + n[1] * p],
						z=[zs[i], zs[i] + n[2] * p],
						name='pts',
						marker=dict(size=1, color="red"),
						mode='lines')
					)

fig = go.Figure(data=plotData)
print('Saving to {0}'.format(os.path.abspath('visualize_cam_light.html')))
fig.write_html('visualize_cam_light.html', auto_open=True)