import os, os.path as op, sys

# DATA = "Lego"
DATA = "bunny_static"
RES = "576x768"
ARCH = "nsvf_base"
SUFFIX = "v1"
# DATASET = "data/Synthetic_NeRF/" + DATA
DATASET = "/home/mazlov/documents/thesis/codes/blender/" + DATA
SAVE = "checkpoint/" + DATA
MODEL = ARCH + SUFFIX
MODEL_PATH = SAVE + '/' + MODEL + '/checkpoint_last.pt'

# SAVE=/checkpoint/jgu/space/neuralrendering/new_release/$DATA

# additional rendering args
CHUNK_SIZE = '256'
RAYMARCHING_TOLERANCE = '0.0'
# MODELARGS = '{"chunk_size":'+CHUNK_SIZE+',"raymarching_tolerance":'+RAYMARCHING_TOLERANCE+',"use_octree":True}'
MODELARGS = ''
# create directory if doesn't exist
# if not os.path.exists(SAVE + '/' + MODEL): os.makedirs(SAVE + '/' + MODEL)

# create configuration file
with open('configuration_render.txt', 'w') as f:
	f.write(DATASET)
	f.write('\n--user-dir fairnr')
	f.write('\n--task single_object_rendering')
	f.write('\n--path '+ MODEL_PATH)
	f.write('\n--render-beam 1')
	f.write('\n--render-save-fps 24')
	f.write('\n--render-camera-poses '+DATASET+'/test_traj.txt')
	f.write('\n--model-overrides '+MODELARGS)
	f.write('\n--render-resolution '+RES)
	f.write('\n--render-output '+SAVE+'/'+ARCH+'/output')
	f.write('\n--render-output-types "color" "depth" "voxel" "normal"')
	f.write('\n--render-combine-output --log-format "simple"')