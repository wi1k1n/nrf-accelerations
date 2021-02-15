import argparse, sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PATH = "D:\\edu\\UniBonn\\Study\\thesis\\codes\\blender\\data\\drums"

filepath = os.path.join(PATH, 'transforms.json')

data = None
with open(filepath, 'r') as f:
	data = json.load(f)

if data is None:
	print('Failed to open transforms.json')
	exit(1)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cams = []
lights = []

# Iterate over renders
for frame in data['frames']:
	rfp = frame['file_path']
	print(rfp)
	Tcam = np.array(frame['transform_matrix'])  # camera
	Tpls = np.array(frame['pl_transform_matrix'])  # point light source

	p = np.array([0, 0, 0, 1])
	cams.append(np.matmul(Tcam, p))
	lights.append(np.matmul(Tpls, p))

cams = np.array(cams)
lights = np.array(lights)

ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], c='blue')
ax.scatter(lights[:, 0], lights[:, 1], lights[:, 2], c='yellow')
plt.show()