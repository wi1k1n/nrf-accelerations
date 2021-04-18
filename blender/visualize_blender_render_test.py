import plotly.graph_objects as go
import pickle
import numpy as np

FILEPATH = 'saved.pckl'

input = pickle.load(open(FILEPATH, 'rb'))

plotData = []

bbox_corners = input['bbox']
conseq = [0, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4]
plotData.append(go.Scatter3d(x=bbox_corners[0, conseq], y=bbox_corners[1, conseq], z=bbox_corners[2, conseq], name='bbox', line=dict(color="red"), mode="lines"))


cams = input['cams']
cam_pts = np.array([[0, 0, 0, 1], [0, 0, -1, 1]])
for cam in cams:
	pts_rot = np.matmul(cam, cam_pts.T)
	plotData.append(go.Scatter3d(x=pts_rot[0], y=pts_rot[1], z=pts_rot[2], name='cam{:2}'.format(ind), line=dict(color="blue", width=3), marker=dict(size=0), mode="lines"))


fig = go.Figure(data=plotData)
fig.write_html('blender_render_test.html')