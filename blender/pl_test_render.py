import sys, os, os.path as op, argparse, re
# print("Python version")
# print (sys.version)
# print (sys.executable)
# print("Version info.")
# print (sys.version_info)
# exit()
import json, time
import bpy
import mathutils
from mathutils import Vector
import numpy as np
from pprint import pprint
from pathlib import Path



CAM_POS = (0, 0, 50)
PL_POS = (0, 0, 5)
PL_POWER = 1000
CAM_LOOKAT = (0, 0, 0)
RESOLUTION = 128
FALLOF_TYPE = 'INVERSE_COEFFICIENTS'



np.random.seed(2)  # fixed seed

OUTDIR = op.abspath(bpy.path.abspath('_pl_test'))
os.makedirs(OUTDIR, exist_ok=True)


### Render Optimizations
bpy.context.scene.render.use_persistent_data = True

### ====== Hardware setup ======
print("---------------   SCENE LIST   ---------------")
for scene in bpy.data.scenes:
    print('Scene name: ', scene.name)
    scene.cycles.device = 'GPU'
    scene.render.resolution_percentage = 200
    scene.cycles.samples = 500
    scene.cycles.max_bounces = 12
print()

# Enable CUDA
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

# Enable and list all devices, or optionally disable CPU
print("---------------   DEVICE LIST   ---------------")
for devices in bpy.context.preferences.addons['cycles'].preferences.get_devices():
    for d in devices:
        d.use = True
        if d.type == 'CPU':
            d.use = False
        print("Device '{}' type {} : {}" . format(d.name, d.type, d.use))
print()


scene = bpy.context.scene

scene.render.filepath = op.join(OUTDIR, 'render')
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.exr_codec = 'PIZ'
scene.render.image_settings.color_depth = '16'

scene.render.dither_intensity = 0.0
scene.render.film_transparent = True


scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100



### Setup camera for rendering
assert scene.objects.get('Camera'), 'Camera not found. Please make sure there is a camera with the name "Camera"'
cam = scene.objects['Camera']

cam.location = CAM_POS

camlookat = bpy.data.objects.new("Empty", None)
camlookat.location = CAM_LOOKAT
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
cam_constraint.target_space = 'WORLD'
cam_constraint.target = camlookat




# Set up the point light to be colocated with camera
if 'PointLight' in scene.objects:
    pointLight = scene.objects['PointLight']
    assert scene.objects.get('PointLight'), 'PointLight not found. Please make sure there is a light with the name "PointLight"'

    pointLight.location = PL_POS

    pointLight.data.energy = PL_POWER


    # print('\n\t'.join('{}: {}'.format(item, getattr(pointLight.data, item)) for item in dir(pointLight.data)))

    # print(pointLight)

    # if 'FALLOF_TYPE' in locals():
    #     pointLight.data.falloff_type = FALLOF_TYPE

    print('energy (?): ', pointLight.data.energy)
    print('falloff_type: ', pointLight.data.falloff_type)
    print('constant_coefficient (k_c): ', pointLight.data.constant_coefficient)
    print('linear_attenuation (?): ', pointLight.data.linear_attenuation)
    print('linear_coefficient (k_l): ', pointLight.data.linear_coefficient)
    print('quadratic_attenuation (?): ', pointLight.data.quadratic_attenuation)
    print('quadratic_coefficient (k_q): ', pointLight.data.quadratic_coefficient)
elif 'SphereLight' in scene.objects:
    sphereLight = scene.objects['SphereLight']

    sphereLight.location = PL_POS

    print('WARNING: Only location is applied to the SphereLight. The strength value has not been changed!!!')
else:
    raise Exception('No light found on the scene!!')






scene.frame_set(scene.frame_current)
bpy.context.view_layer.update()


print('BEFORE RENDER')
# exit('Quitting...')
bpy.ops.render.render(write_still=True)  # render still
print('AFTER RENDER')