import bpy
import numpy as np
from mathutils import Vector
import pickle

print('========================================')

VIEWS = 500
CAM_HEMISPHERE_ANGLES = [0, 90]
CAM_DISTANCE = 10
FILEPATH = 'saved.pckl'


output = {}

obj = bpy.context.scene.objects['Cube']
origin = obj.location
print('Cube location: ', origin)


bbox_corners = np.asarray([obj.matrix_world @ Vector(corner) for corner in obj.bound_box]).T
output['bbox'] = bbox_corners  # 3 x 8

cam = bpy.context.scene.objects['Camera']

camDst = CAM_DISTANCE
camAngles = (min(CAM_HEMISPHERE_ANGLES), max(CAM_HEMISPHERE_ANGLES))
print('Angles: ', CAM_HEMISPHERE_ANGLES)

camAngles = np.radians(camAngles)
camAngDiff = camAngles[1] - camAngles[0]
camAnglesSin, camAnglesCos = np.sin(camAngles), np.cos(camAngles)
maxCos = max(camAnglesCos) if np.sign(camAnglesSin[0]) == np.sign(camAnglesSin[1]) else 1
cam.location = (origin[0], camDst * maxCos + origin[1], camDst * np.sin(np.arccos(maxCos)) + origin[2])  # !!wrong, if camAngles[1] < 0, because arrcos
print('Camera location: ', cam.location)

b_empty = bpy.data.objects.new("Empty", None)
b_empty.location = origin
bpy.context.scene.collection.objects.link(b_empty)
bpy.context.view_layer.objects.active = b_empty
cam.parent = b_empty  # setup parenting

camMats = []
for i in range(0, VIEWS):
	rot = np.random.uniform(0, np.pi, size=3) * (0, 1.5, 0)
	b_empty.rotation_euler = rot
	bpy.context.view_layer.update()
	camMats.append(cam.matrix_world)

output['cams'] = camMats

pickle.dump(output, open(FILEPATH, "wb"))