import argparse, sys, os, json
import numpy as np

import plotly.graph_objects as go

PATH = "D:\\edu\\UniBonn\\Study\\thesis\\codes\\blender\\datasets\\guitar"

transformsPath = os.path.join(PATH, 'transforms.json')
bboxPath = os.path.join(PATH, 'bbox.txt')

data = None
with open(transformsPath, 'r') as f:
	data = json.load(f)

if data is None:
	print('Failed to open transforms.json')
	exit(1)

# Read bounding box for better visualization
bbox = None
if os.path.isfile(bboxPath):
	bbox = np.loadtxt(bboxPath)
if bbox is not None and len(bbox) == 7:
	xn, yn, zn, xx, yx, zx, _ = tuple(bbox)
	#           0   1   2   3   0   4   5   1   5   6   2   6   7   3   7   4
	cornersX = [xn, xn, xx, xx, xn, xn, xn, xn, xn, xx, xx, xx, xx, xx, xx, xn]
	cornersY = [yx, yn, yn, yx, yx, yx, yn, yn, yn, yn, yn, yn, yx, yx, yx, yx]
	cornersZ = [zn, zn, zn, zn, zn, zx, zx, zn, zx, zx, zn, zx, zx, zn, zx, zx]

cams = []
lights = []
dirs = []

# Iterate over renders
for frame in data['frames']:
	rfp = frame['file_path']
	print(rfp)
	Tcam = np.array(frame['transform_matrix'])  # camera
	Tpls = np.array(frame['pl_transform_matrix'])  # point light source

	p = np.array([0, 0, 0, 1])
	q = np.array([0, 0, -0.2, 1])
	cams.append(np.matmul(Tcam, p))
	dirs.append(np.matmul(Tcam, q))
	lights.append(np.matmul(Tpls, p))

cams = np.array(cams)
lights = np.array(lights)

# light positions
data = [
	# go.Scatter3d(x=cams[:, 0], y=cams[:, 1], z=cams[:, 2], marker=go.scatter3d.Marker(size = 3), mode='markers'),
	go.Scatter3d(x=lights[:, 0], y=lights[:, 1], z=lights[:, 2], name='lights', marker=dict(size=5, color="orange"), mode='markers'),
]

# add bounding box
if bbox is not None and len(bbox) == 7:
	data.append(go.Scatter3d(x=cornersX, y=cornersY, z=cornersZ, name='bbox', mode='lines'))

# add camera positions
for ind, dir in enumerate(dirs):
	data.append(go.Scatter3d(x=[cams[ind, 0], dir[0]], y=[cams[ind, 1], dir[1]], z=[cams[ind, 2], dir[2]], name='cam{:2}'.format(ind), line=dict(color="blue", width=3), marker=dict(size=0), mode="lines"))


fig = go.Figure(data=data)
fig.write_html('visualize_cam_light.html', auto_open=True)