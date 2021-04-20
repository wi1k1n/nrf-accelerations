import plotly.graph_objects as go
import pickle
import numpy as np

plotData = []

bbox_corners = np.load('bbox.npy')
conseq = [0, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4]
plotData.append(go.Scatter3d(x=bbox_corners[0, conseq], y=bbox_corners[1, conseq], z=bbox_corners[2, conseq], name='bbox', line=dict(color="red"), mode="lines"))


cams = np.load('cams.npy')
# cam_pts = np.array([[0, 0, 0, 1], [0, 0, -0.1, 1]])
cam_pts = np.array([0, 0, 0, 1])
cam_ptss = []
for ind in range(cams.shape[0]):
	cam = cams[ind, :, :]
	pts_rot = np.matmul(cam, cam_pts.T)
	# plotData.append(go.Scatter3d(x=pts_rot[0], y=pts_rot[1], z=pts_rot[2], name='cam{:2}'.format(ind), line=dict(color="blue", width=3), marker=dict(size=0), mode="lines"))
	cam_ptss.append(pts_rot)
cam_ptss = np.array(cam_ptss).T

plotData.append(go.Scatter3d(x=cam_ptss[0], y=cam_ptss[1], z=cam_ptss[2], name='cams', marker=dict(size=3, color='orange'), mode="markers"))

scene=dict(xaxis=dict(),
			yaxis=dict(),
			zaxis=dict(),
			aspectmode='auto', #this string can be 'data', 'cube', 'auto', 'manual'
			#a custom aspectratio is defined as follows:
			aspectratio=dict(x=1, y=1, z=1)
			)

fig = go.Figure(data=plotData, layout=go.Layout(scene=scene))
fig.write_html('blender_render_test.html')