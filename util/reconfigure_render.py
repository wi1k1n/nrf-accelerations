import os, os.path as op, sys
from reconfigure_utils import inject_pycharm_config
import pyperclip


COPY2CLIPBOARD = False
INJECT_PYCHARM = True
SAVE_FILE = True

DATA = "rocket_coloc_png"
NAME = "test"  # postfix for dataset name
RES = "128x128"

CHUNK_SIZE = '256'

WITH_LIGHT = True
ARCH = "mlnrf_base"# ARCH = "nsvf_base"

SUFFIX = "v1"
DATASET = "/home/mazlov/documents/thesis/codes/blender/" + DATA  # "data/Synthetic_NeRF/" + DATA
SAVE = "checkpoint/" + DATA + (('_' + NAME) if NAME else '')
MODEL = ARCH + SUFFIX
MODEL_PATH = SAVE + '/' + MODEL + '/checkpoint_last.pt'

RAYMARCHING_TOLERANCE = '0.01'
# MODELARGS = '{"chunk_size":'+CHUNK_SIZE+',"raymarching_tolerance":'+RAYMARCHING_TOLERANCE+'}'
MODELARGS = ''


HALF_VOXEL_SIZE_AT = '5000,12500'#,35000'  # '5000,25000,75000'
REDUCE_STEP_SIZE_AT = '2000,8500,35000'  # '5000,25000,75000'
PRUNNING_EVERY_STEPS = '2500'#'5000'  # '1500'
SAVE_INTERVAL_UPDATES = '500'#'750'  # '100'
TOTAL_NUM_UPDATE = '75000'  # 150000

# PREPROCESS = 'none'  # none/mstd/minmax
# MIN_COLOR = '-1'  # '-1'
# BG_COLOR = '1.0,1.0,1.0'  # '1.0,1.0,1.0'


XML_PATH = '.run/render.run.xml'
NUM_BACKUPS = 10

# create configuration file
parameters = ''
parameters += DATASET
parameters += '\n--path ' + MODEL_PATH
# parameters += '\n--model-overrides \'{"chunk_size":'+CHUNK_SIZE+',"raymarching_tolerance":0.01}\''
parameters += '\n--render-beam 1'
parameters += '\n--render-save-fps 24'
# parameters += '\n--render-camera-poses ' + DATASET + '/test_traj.txt'
if len(MODELARGS):
	parameters += '\n--model-overrides ' + MODELARGS
parameters += '\n--render-resolution ' + RES
parameters += '\n--render-output ' + SAVE + '/' + ARCH + '/output'
parameters += '\n--render-output-types "color" "depth" "voxel" "normal"'
parameters += '\n--render-combine-output --log-format "simple"'
# parameters += '\n--initial-boundingbox ' + DATASET + '/bbox.txt'
parameters += '\n--seed 2'
parameters += '\n--log-format simple'
parameters += '\n--log-interval 1'


# python render.py ${DATASET} \
#     --model-overrides '{"chunk_size":512,"raymarching_tolerance":0.01}' \
#     --render-save-fps 24 \
#     --render-resolution "800x800" \
#     --render-camera-poses ${DATASET}/pose \
#     --render-views "200..400" \
#     --render-output ${SAVE}/output \
#     --render-output-types "color" "depth" "voxel" "normal" --render-combine-output \
#     --log-format "simple"

if SAVE_FILE:
	with open('configuration_render.txt', 'w') as f:
		f.write(parameters)

if COPY2CLIPBOARD:
	pyperclip.copy(parameters)

if INJECT_PYCHARM:
	inject_pycharm_config('render', XML_PATH, parameters, NUM_BACKUPS)





######################################################################
######################################################################
######################################################################

# # DATA = "Lego"
# DATA = "bunny_static"
# RES = "576x768"
# ARCH = "nsvf_base"
# SUFFIX = "v1"
# # DATASET = "data/Synthetic_NeRF/" + DATA
# DATASET = "/home/mazlov/documents/thesis/codes/blender/" + DATA
# SAVE = "checkpoint/" + DATA
# MODEL = ARCH + SUFFIX
# MODEL_PATH = SAVE + '/' + MODEL + '/checkpoint_last.pt'
#
# # SAVE=/checkpoint/jgu/space/neuralrendering/new_release/$DATA
#
# # additional rendering args
# CHUNK_SIZE = '256'
# RAYMARCHING_TOLERANCE = '0.0'
# # MODELARGS = '{"chunk_size":'+CHUNK_SIZE+',"raymarching_tolerance":'+RAYMARCHING_TOLERANCE+',"use_octree":True}'
# MODELARGS = ''
# # create directory if doesn't exist
# # if not os.path.exists(SAVE + '/' + MODEL): os.makedirs(SAVE + '/' + MODEL)
#
# # create configuration file
# with open('configuration_render.txt', 'w') as f:
# 	f.write(DATASET)
# 	f.write('\n--user-dir fairnr')
# 	f.write('\n--task single_object_rendering')
# 	f.write('\n--path '+ MODEL_PATH)
# 	f.write('\n--render-beam 1')
# 	f.write('\n--render-save-fps 24')
# 	f.write('\n--render-camera-poses '+DATASET+'/test_traj.txt')
# 	f.write('\n--model-overrides '+MODELARGS)
# 	f.write('\n--render-resolution '+RES)
# 	f.write('\n--render-output '+SAVE+'/'+ARCH+'/output')
# 	f.write('\n--render-output-types "color" "depth" "voxel" "normal"')
# 	f.write('\n--render-combine-output --log-format "simple"')