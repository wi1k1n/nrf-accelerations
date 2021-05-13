import argparse, sys, os, json, subprocess
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

MODEL = 'rocket_test'
PATH = 'D:\\edu\\UniBonn\\Study\\thesis\\codes\\blender\\datasets\\' + MODEL
CAM_ICONS = False



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
else:
	print('Failed to open bbox.txt. Continuing without bounding box.')
if bbox is not None and len(bbox) == 7:
	xn, yn, zn, xx, yx, zx, _ = tuple(bbox)
	#           0   1   2   3   0   4   5   1   5   6   2   6   7   3   7   4
	cornersX = [xn, xn, xx, xx, xn, xn, xn, xn, xn, xx, xx, xx, xx, xx, xx, xn]
	cornersY = [yx, yn, yn, yx, yx, yx, yn, yn, yn, yn, yn, yn, yx, yx, yx, yx]
	cornersZ = [zn, zn, zn, zn, zn, zx, zx, zn, zx, zx, zn, zx, zx, zn, zx, zx]



# reading stl/obj/ply model
mesh = None  # (vertices, I, J, K)
# looking for model file and convert it from stl, if needed
for ext in ['stl', 'ply', 'obj']:
	path_mesh = os.path.join(PATH, MODEL + '.' + ext)
	if os.path.isfile(path_mesh):
		mesh = o3d.io.read_triangle_mesh(path_mesh)
if mesh:
	vertices = np.asarray(mesh.vertices).T
	triangles = np.asarray(mesh.triangles).T
else:
	print('Failed to open model file. Only PLY, OBJ and STL files are supported. Continuing without model.')



cams = []
lights = []
dirs = []

# Iterate over renders
for frame in data['frames']:
	rfp = frame['file_path']
	# print(rfp)
	Tcam = np.array(frame['transform_matrix'])  # camera
	Tpls = np.array(frame['pl_transform_matrix'])  # point light source

	p = np.array([0, 0, 0, 1])
	q = np.array([0, 0, -0.2, 1])
	cams.append(np.matmul(Tcam, p))
	dirs.append(np.matmul(Tcam, q))
	lights.append(np.matmul(Tpls, p))

if not len(cams):
	print('No camera data found. Stopping...')
	exit()

cams = np.array(cams)
lights = np.array(lights)

plotData = []

# light positions
light_markers = [dict(size=3, color="orange", symbol='x'), dict(size=7, color="orange", symbol='cross')]
if len(lights):
	[plotData.append(go.Scatter3d(x=lights[:, 0], y=lights[:, 1], z=lights[:, 2],
								 name='lights',
								 marker=m,
								 mode='markers')) \
	for i, m in enumerate(light_markers)]

# add bounding box
if bbox is not None and len(bbox) == 7:
	plotData.append(go.Scatter3d(x=cornersX, y=cornersY, z=cornersZ, name='bbox', mode='lines'))

# add model
if mesh is not None:
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



# camera icon sizes
# camScale = 1.5 * 0.01
camScale = 1e-2 * np.array((bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2])).mean()
if CAM_ICONS:
	bw, bh, bt, lw, lh, lt = 1, 1, 3, 4.5, 4.5, 2.5

	dw, dh = 0.5 * (lw - bw), 0.5 * (lh - bh)
	tt = bt + lt
	pts_x = np.array([-bw, -bw, bw, bw, -bw, -bw, bw, bw, -bw - dw, -bw - dw, bw + dw, bw + dw]) * camScale
	pts_y = np.array([-bh, bh, bh, -bh, -bh, bh, bh, -bh, -bh - dh, bh + dh, bh + dh, -bh - dh]) * camScale
	pts_z = -np.array([-bt, -bt, -bt, -bt, 0, 0, 0, 0, lt, lt, lt, lt]) * camScale
	conseq = [0, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4, 8, 9, 5, 9, 10, 6, 10, 11, 7, 11, 8]

	cam_pts = np.array([p for p in zip(pts_x[conseq], pts_y[conseq], pts_z[conseq], np.ones(len(conseq)))])
else:
	cam_pts = np.array([[0, 0, 0, 1], [0, 0, -10 * camScale, 1]])

# add camera positions
for ind, dir in enumerate(dirs):
	Tpls = np.array(data['frames'][ind]['transform_matrix'])
	pts_rot = np.matmul(Tpls, cam_pts.T)
	plotData.append(go.Scatter3d(x=pts_rot[0], y=pts_rot[1], z=pts_rot[2], name='cam{:2}'.format(ind), line=dict(color="blue", width=3), marker=dict(size=0), mode="lines"))



fig = go.Figure(data=plotData)
print('Saving to {0}'.format(os.path.abspath('visualize_cam_light.html')))
fig.write_html('visualize_cam_light.html', auto_open=True)
print('Saved. File size: {0} bytes'.format(os.path.getsize('visualize_cam_light.html')))