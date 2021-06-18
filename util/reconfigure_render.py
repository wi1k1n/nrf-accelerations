import os, os.path as op, sys
from reconfigure_utils import inject_pycharm_config
import pyperclip


COPY2CLIPBOARD = False
INJECT_PYCHARM = True
SAVE_FILE = True

DATA = "rocket_random_png"
NAME = ""  # postfix for dataset name
RENDER_OUTPUT = "output_test"  # output if empty
RES = "256x256"
RENDER_PATH_LIGHT = False  # True - light source is moving, false - camera is moving
NUM_FRAMES = '180'
TARGETS_PATH = '/data/mazlov2/Documents/thesis/codes/blender/' + DATA + '_target_' + ('light' if RENDER_PATH_LIGHT else 'cam') + '/target'
DRY_RUN = False  # only create camera/light positions and do not evaluate model

CHUNK_SIZE = '256'
RENDER_BEAM = '4'  # should be an even divisor of NUM_FRAMES TODO: fix it

# WITH_LIGHT = True
ARCH = "mlnrf_base"
TASK = 'single_object_light_rendering'
# ARCH = "nsvf_base"
# TASK = 'single_object_rendering'

SUFFIX = "v1"
DATASET = "datasets/" + DATA  # "data/Synthetic_NeRF/" + DATA
SAVE = "checkpoint/" + DATA + (('_' + NAME) if NAME else '')
MODEL = ARCH + SUFFIX
MODEL_PATH = SAVE + '/' + MODEL + '/checkpoint_last.pt'

## Rocket
# RENDER_PATH_ARGS = '{\'radius\':1.5,\'h\':3,\'o\':(-0.1,0.05,1.25)}'  # top diagonal view
RENDER_PATH_ARGS = '{\'radius\':4.5,\'h\':0.0,\'o\':(-0.1,0.05,1.25)}'
RENDER_AT_VECTOR = '"(-0.1,0.05,1.25)"'
## Guitar
# RENDER_PATH_ARGS = '{\'radius\':0.8,\'h\':1.0,\'o\':(0,-0.06,0.5)}'
# RENDER_AT_VECTOR = '"(0, -0.06, 0.575)"'

RENDER_PATH_STYLE = 'circle'
RENDER_SPEED = '2'


RAYMARCHING_TOLERANCE = '0.01'
# MODELARGS = '{"chunk_size":'+CHUNK_SIZE+',"raymarching_tolerance":'+RAYMARCHING_TOLERANCE+'}'
MODELARGS = ''
NUM_WORKERS = ''

# PREPROCESS = 'none'  # none/mstd/minmax
# MIN_COLOR = '-1'  # '-1'
# BG_COLOR = '1.0,1.0,1.0'  # '1.0,1.0,1.0'

XML_PATH = '.run/render.run.xml'
NUM_BACKUPS = 10




# create configuration file
parameters = ''
parameters += DATASET
parameters += '\n--path ' + MODEL_PATH
parameters += '\n--task ' + TASK
if not RENDER_OUTPUT: RENDER_OUTPUT = "output"
parameters += '\n--render-output ' + SAVE + '/' + ARCH + '/' + RENDER_OUTPUT + ('_light' if RENDER_PATH_LIGHT else '_cam')
parameters += '\n--render-path-style ' + RENDER_PATH_STYLE
parameters += '\n--render-path-args ' + RENDER_PATH_ARGS
parameters += '\n--render-at-vector ' + RENDER_AT_VECTOR
parameters += '\n--render-angular-speed ' + RENDER_SPEED
# if WITH_LIGHT:
# 	parameters += '\n--with-point-light'
if RENDER_PATH_LIGHT:
	parameters += '\n--render-path-light'
parameters += '\n--render-num-frames ' + NUM_FRAMES
if TARGETS_PATH:
	parameters += '\n--targets-path ' + TARGETS_PATH
if DRY_RUN:
	parameters += '\n--render-dry-run'

# parameters += '\n--model-overrides \'{"chunk_size":'+CHUNK_SIZE+',"raymarching_tolerance":0.01}\''
parameters += '\n--render-beam ' + RENDER_BEAM
parameters += '\n--render-save-fps 24'
# parameters += '\n--render-camera-poses ' + DATASET + '/test_traj.txt'
if len(MODELARGS):
	parameters += '\n--model-overrides ' + MODELARGS
parameters += '\n--render-resolution ' + RES
# parameters += '\n--render-output ' + SAVE + '/' + ARCH + '/output'
parameters += '\n--render-output-types ' + ('"target"' if TARGETS_PATH else '') + ' "color" "depth" "voxel" "normal"'
parameters += '\n--render-combine-output --log-format "simple"'
# parameters += '\n--initial-boundingbox ' + DATASET + '/bbox.txt'
if NUM_WORKERS:
	parameters += '\n--num-workers ' + NUM_WORKERS
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