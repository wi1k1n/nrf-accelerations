# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys, os, argparse
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

sys.path.append(os.getcwd())
from render_params import opts


np.random.seed(2)  # fixed seed
POSTPROCESSING_SCRIPT = True


### Argument Parser code
parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('output', type=str, help='path where files will be saved')

argv = sys.argv
argv = argv[argv.index("--") + 1:]
args = parser.parse_args(argv)

### Manage directories
homedir = args.output
fp = bpy.path.abspath(f"{homedir}/{opts.RESULTS_PATH}")

print('\n\n>> Processing model ' + os.path.abspath(opts.PATH_MODEL))
print('>> Into folder ' + os.path.abspath(bpy.path.abspath(f"{homedir}")) + '\n\n')

if not os.path.exists(fp):
    os.makedirs(fp)
if not os.path.exists(os.path.join(homedir, "pose")):
    os.mkdir(os.path.join(homedir, "pose"))
if not os.path.exists(os.path.join(homedir, "pose_pl")):
    os.mkdir(os.path.join(homedir, "pose_pl"))

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}

### Render Optimizations
bpy.context.scene.render.use_persistent_data = True


### ====== Hardware setup ======
print("---------------   SCENE LIST   ---------------")
for scene in bpy.data.scenes:
    print('Scene name: ', scene.name)
    scene.cycles.device = 'GPU'
    scene.render.resolution_percentage = 200
    scene.cycles.samples = opts.CYCLES_SAMPLES
    scene.cycles.max_bounces = opts.CYCLES_MAX_BOUNCES
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

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
#bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(opts.FORMAT)
if (opts.FORMAT == 'OPEN_EXR'): bpy.context.scene.render.image_settings.exr_codec = 'PIZ'
bpy.context.scene.render.image_settings.color_depth = str(opts.COLOR_DEPTH)

if not opts.DEBUG:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    if opts.FORMAT == 'OPEN_EXR':
      links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
      # Remap as other types can not represent the full range of depth.
      map = tree.nodes.new(type="CompositorNodeMapValue")
      # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
      map.offset = [-0.7]
      map.size = [opts.DEPTH_SCALE]
      map.use_min = True
      map.min = [0]
      links.new(render_layers.outputs['Depth'], map.inputs[0])

      links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

### Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background
objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})





### Calculate bounding box size
bbox = None

# First check collections to contain collection 'Shape'
print("---------------   COLLECTION LIST   ---------------")
for collection in bpy.data.collections:
    print(collection.name)
print()
collShape = bpy.data.collections.get('Shape')
objShape = []
if collShape:
    for o in collShape.all_objects:
        objShape.append(o)
else:
    print('Collection "Shape" not found, looking for object "Shape"...')
    # Try to find object with name 'Shape' if collection was not found
    print("---------------   OBJECT LIST   ---------------")
    for obj in bpy.context.scene.objects:
        print(obj.name)
    print()

    oShape = bpy.context.scene.objects.get('Shape')
    assert oShape, 'Object "Shape" not found. Check that there is a collection or object with name "Shape"'
    objShape.append(oShape)

# Iterate over all shapes to create the whole bounding box
print("---------------   OBJECT LIST (for bbox)   ---------------")
for obj in objShape:
    assert obj.type in ('MESH', 'SURFACE'), 'Invalid object type \'' + obj.type + '\'! Please make sure to have only MESH and SURFACE objects!'
    print('(' + obj.type + ') ' + obj.name)

    # print([['{:.5f}'.format(f) for f in (obj.matrix_world @ Vector(corner))] for corner in obj.bound_box])
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    if not bbox: bbox = corners
    # print(corners)
    bbox = [min([cc[i] for cc in corners] + [bbox[i]]) for i in range(3)] + \
           [max([cc[i] for cc in corners] + [bbox[i+3]]) for i in range(3)]
    # print(" ".join(['{:.5f}'.format(f) for f in bbox]))
print()

voxel_size = ((bbox[3]-bbox[0]) * (bbox[4]-bbox[1]) * (bbox[5]-bbox[2]) / opts.VOXEL_NUMS) ** (1/3)
# bbox format: x_min y_min z_min x_max y_max z_max initial_voxel_size
print(" ".join(['{:.5f}'.format(f) for f in bbox + [voxel_size]]),
    file=open(os.path.join(homedir, 'bbox.txt'), 'w'))
print('bbox: ', " ".join(['{:.5f}'.format(f) for f in bbox + [voxel_size]]))
assert bbox is not None, 'bbox.txt is not created. Check the main shape to have name "Shape"'


## DEBUG
# cb = bpy.ops.mesh.primitive_cube_add(size=1.1, location=((bbox[0]+bbox[3])*0.5, (bbox[1]+bbox[4])*0.5, (bbox[2]+bbox[5])*0.5))


modelCenter = ((bbox[0]+bbox[3])*0.5, (bbox[1]+bbox[4])*0.5, (bbox[2]+bbox[5])*0.5)
def parent_obj_to_camera(b_camera):
    # origin = (0, 0, 0)
    origin = modelCenter
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = opts.RESOLUTION
scene.render.resolution_y = opts.RESOLUTION
scene.render.resolution_percentage = 100




### Setup camera for rendering
assert scene.objects.get('Camera'), 'Camera not found. Please make sure there is a camera with the name "Camera"'
cam = scene.objects['Camera']
# Use overrided cam_distance from render parameters if exists
camDst = None
if hasattr(opts, 'CAM_DISTANCE') and opts.CAM_DISTANCE:
    camDst = opts.CAM_DISTANCE
else:
    # otherwise try using custom property in .blend file
    camDst = bpy.context.scene.get('camDst')
assert camDst is not None, 'Neither CAM_DISTANCE parameter nor camDst custom property specified. Please either setup camera distance to scene property in blender project file or override in render_params.py'

camAngles = (min(opts.CAM_HEMISPHERE_ANGLES), max(min(opts.CAM_HEMISPHERE_ANGLES)))
assert camAngles[0] >= -90 and camAngles[0] < camAngles[1] and camAngles[1] <= 90, 'CAM_HEMISPHERE_ANGLES should be a list of [a_min, a_max] angles, calculated from XY plane'

camAngles = np.asarray(camAngles) * np.pi / 180.0
camAnglesSin, camAnglesCos = np.sin(camAngles), np.cos(camAngles)
maxCos = 1 if np.sign(camAnglesSin[0]) and np.sign(camAnglesSin[1]) else max(camAngleCos)
cam.location = (0, camDst * maxCos, camDst * np.sin(np.acos(maxCos)) + modelCenter[2])
if not opts.RANDOM_VIEWS:
    cam.location[2] += camDst * camAnglesSin[1]
# cam.location = (0, 0, 15)
# cam.location = (0, camDst)
print('Camera initial location: ', cam.location)

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
cam_constraint.target_space = 'WORLD'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

# Set up the point light to be colocated with camera
pointLight = scene.objects['PointLight']
assert scene.objects.get('PointLight'), 'PointLight not found. Please make sure there is a light with the name "PointLight"'
if opts.LIGHT_SETUP == 'none':
    bpy.ops.object.delete({"selected_objects": [pointLight]})
elif opts.LIGHT_SETUP == 'colocated':
    # pointLight.location = b_empty.location
    pointLight.parent = b_empty
    pointLight.location = cam.location

# scene.render.image_settings.file_format = 'PNG'  # set output format to .png

from math import radians

stepsize = 360.0 / opts.VIEWS
# rotation_mode = 'XYZ'

if not opts.DEBUG:
    for output_node in [depth_file_output, normal_file_output]:
        output_node.base_path = ''

out_data['frames'] = []

# if not opts.RANDOM_VIEWS:
#     b_empty.rotation_euler = CIRCLE_FIXED_START

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

# Iterating over views
for i in range(0, opts.VIEWS):
    # if opts.DEBUG:
    #     i = np.random.randint(0, opts.VIEWS)
    #     b_empty.rotation_euler[2] += radians(stepsize*i)
    scene.render.filepath = os.path.join(fp, '{:04d}'.format(i))
    if opts.RANDOM_VIEWS:
        rot = np.random.uniform(0, np.pi, size=3) * (2, -1, 0)
        # rot[0] = np.abs(rot[0])
        b_empty.rotation_euler = rot
        # if opts.UPPER_VIEWS:
        #     # rot = np.random.uniform(0, 1, size=3) * (2, 0, 2 * np.pi)
        #     # rot[0] = np.abs(np.arccos(1 - rot[0]) - np.pi/2)
        #     # b_empty.rotation_euler = rot
        #     rot = np.random.uniform(0, np.pi, size=3) * (2, -1, 0)
        #     # rot[0] = np.abs(rot[0])
        #     b_empty.rotation_euler = rot
        # else:
        #     b_empty.rotation_euler = np.random.uniform(0, 2 * np.pi, size=3)
    else:
        print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))

    # apply changes
    bpy.context.view_layer.update()

    # print(b_empty.rotation_euler, cam.matrix_world)

    # depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
    # normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

    frame_data = {
        'file_path': scene.render.filepath,
        'rotation': radians(stepsize),
        'transform_matrix': listify_matrix(cam.matrix_world),
        'pl_transform_matrix': listify_matrix(pointLight.matrix_world)
    }
    out_data['frames'].append(frame_data)

    with open(os.path.join(homedir, "pose", '{:04d}.txt'.format(i)), 'w') as fo:
        for ii, pose in enumerate(frame_data['transform_matrix']):
            print(" ".join([str(-p) if (((j == 2) | (j == 1)) and (ii < 3)) else str(p)
                            for j, p in enumerate(pose)]),
                file=fo)
    with open(os.path.join(homedir, "pose_pl", '{:04d}.txt'.format(i)), 'w') as fo:
        for ii, pose in enumerate(frame_data['pl_transform_matrix']):
            print(" ".join([str(-p) if (((j == 2) | (j == 1)) and (ii < 3)) else str(p)
                            for j, p in enumerate(pose)]),
                file=fo)

    if opts.RANDOM_VIEWS:
        if opts.UPPER_VIEWS:
            rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
            rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
            b_empty.rotation_euler = rot
        else:
            b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
    else:
        b_empty.rotation_euler[2] += radians(stepsize)


    if opts.DEBUG:
        continue
    else:
        print('BEFORE RENDER')
        bpy.ops.render.render(write_still=True)  # render still
    print('AFTER RENDER')

if not opts.DEBUG or True:
    with open(os.path.join(homedir, 'transforms.json'), 'w') as out_file:
        json.dump(out_data, out_file, indent=4)


# save camera data
H, W = opts.RESOLUTION, opts.RESOLUTION
f = .5 * W /np.tan(.5 * float(out_data['camera_angle_x']))
cx = cy = W // 2

# write intrinsics
with open(os.path.join(homedir, 'intrinsics.txt'), 'w') as fi:
    print("{} {} {} 0.".format(f, cx, cy), file=fi)
    print("0. 0. 0.", file=fi)
    print("0.", file=fi)
    print("1.", file=fi)
    print("{} {}".format(H, W), file=fi)

# write rendering options
with open(os.path.join(homedir, 'render_params.txt'), 'w') as fi:
    print(opts, file=fi)


# Postprocessing script for the dataset
if POSTPROCESSING_SCRIPT:
    os.system("python postprocess_dataset.py " + fp + " " + homedir)