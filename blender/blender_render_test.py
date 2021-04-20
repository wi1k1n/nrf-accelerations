import bpy
import numpy as np
from mathutils import Vector
import pickle

print('========================================')

VIEWS = 500
CAM_HEMISPHERE_ANGLES = [-5, 75]  # [-90, 90]
CAM_DISTANCE = 10

phi = CAM_HEMISPHERE_ANGLES.copy()
assert phi[0] >= -90 and phi[1] <= 90 and phi[1] > phi[0], ''
print('Phi: ', phi)

obj = bpy.context.scene.objects['Cube']
origin = obj.location
print('Cube location: ', origin)


bbox_corners = np.asarray([obj.matrix_world @ Vector(corner) for corner in obj.bound_box]).T

cam = bpy.context.scene.objects['Camera']

camDst = CAM_DISTANCE
# camAngles = (min(CAM_HEMISPHERE_ANGLES), max(CAM_HEMISPHERE_ANGLES))
# print('Angles: ', CAM_HEMISPHERE_ANGLES)

# camAngles = np.radians(camAngles)
# camAngDiff = camAngles[1] - camAngles[0]
# camAnglesSin, camAnglesCos = np.sin(camAngles), np.cos(camAngles)
# maxCos = max(camAnglesCos) if np.sign(camAnglesSin[0]) == np.sign(camAnglesSin[1]) else 1
# cam.location = (origin[0], camDst * maxCos + origin[1], camDst * np.sin(np.arccos(maxCos)) + origin[2])  # !!wrong, if camAngles[1] < 0, because arrcos
print('Camera location: ', cam.location)

camMats = []
for i in range(0, VIEWS):
	rot = np.random.uniform(0, 1, size=3) * (1 , 1, 0)
	rot[0] = rot[0] * (phi[1] - phi[0]) / 180.0 + (90 + phi[0]) / 180.0
	rot[0] = np.arccos(rot[0] * 2 - 1)
	rot[1] = rot[1] * 2 * np.pi

	cam.location[0] = camDst * np.sin(rot[0]) * np.cos(rot[1]) + origin[0]
	cam.location[1] = camDst * np.sin(rot[0]) * np.sin(rot[1]) + origin[1]
	cam.location[2] = camDst * np.cos(rot[0]) + origin[2]

	bpy.context.view_layer.update()
	# print(cam.matrix_world)
	camMats.append(np.asarray(cam.matrix_world))

# print('=============')
cams = np.array(camMats)
# print(cams)

np.save('bbox', bbox_corners)
np.save('cams', cams)