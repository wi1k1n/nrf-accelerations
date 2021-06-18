import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# mpl.use('TkAgg')
mpl.use('Qt5Agg')



sampled_xyz = np.load('sampled_xyz.npy')
sampled_dir = np.load('sampled_dir.npy')
light_dirs = np.load('light_dirs.npy')



# rdirs = np.loadtxt('ray_dir')
# rstarts = np.loadtxt('ray_start')
#
# x, y, z = zip(*rstarts[:80,...])
# u, v, w = zip(*rdirs[:80,...])

fig = plt.figure()
ax = fig.gca(projection='3d')

# ax.quiver(x, y, z, u, v, w, length=0.2, normalize=True)
v = 0
ax.scatter(sampled_xyz[v,:5,0], sampled_xyz[v,:5,1], sampled_xyz[v,:5,2])
ax.quiver(sampled_xyz[v,:5,0], sampled_xyz[v,:5,1], sampled_xyz[v,:5,2], light_dirs[v,:5,0], light_dirs[v,:5,1], light_dirs[v,:5,2])

plt.show()