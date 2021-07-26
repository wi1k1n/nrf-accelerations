import os, os.path as op, sys
from reconfigure_utils import inject_pycharm_config
import pyperclip


COPY2CLIPBOARD = False
INJECT_PYCHARM = True
SAVE_FILE = True

DATA = "rocket_coloc_exr"
NAME = "test2"  # postfix for dataset name
RENDER_OUTPUT = "output"  # output if empty
RES = "128x128"
RENDER_PATH_LIGHT = True  # True - light source is moving, false - camera is moving
NUM_FRAMES = '180'
TARGETS_PATH = '/data/mazlov2/Documents/thesis/codes/blender/'\
			   + DATA + '_' + NAME + '_target_' + ('light' if RENDER_PATH_LIGHT else 'cam') + '/target'
DRY_RUN = False  # only create camera/light positions and do not evaluate model

CHUNK_SIZE = '256'
RENDER_BEAM = '1'  # should be an even divisor of NUM_FRAMES TODO: fix it

MIN_COLOR = '0.0'  #
MAX_COLOR = '0.8'
BG_COLOR = '0.0'  # '0.25,0.25,0.25'  # '1.0,1.0,1.0'

# WITH_LIGHT = True
# ARCH = "mlnrf_base"
# TASK = 'single_object_light_rendering'
# ARCH = "nsvf_base"
# TASK = 'single_object_rendering'


# # <!-- Original NSVF from facebook -->
# ARCH = "nsvf_base"
# TASK = 'single_object_rendering'
# # <!/-- Original NSVF from facebook -->

# # <!-- Implicit model with ignoring light interaction -->
# ARCH = "mlnrf_base"
# TASK = 'single_object_light_rendering'
# # <!/-- Implicit model with ignoring light interaction -->

# # <!-- Implicit model with InVoxelApproximation light interaction -->
# ARCH = "mlnrfiva_base"
# TASK = 'single_object_light_rendering'
# # <!/-- Implicit model with InVoxelApproximation light interaction -->

# # <!-- Explicit model with ignoring light interaction -->
# ARCH = "mlnrfex_base"
# TASK = 'single_object_light_rendering'
# TRACE_NORMAL = True
# LAMBERT_ONLY = False
# # <!/-- Explicit model with ignoring light interaction -->

# <!-- Explicit model with NRF (colocated!) light interaction -->
ARCH = "mlnrfnrf_base"
TASK = 'single_object_light_rendering'
LAMBERT_ONLY = False
# LIGHT_INTENSITY = '10000.0'
# TEXTURE_LAYERS = '3'
# <!/-- Explicit model with ignoring light interaction -->


DATASET = "datasets/" + DATA  # "data/Synthetic_NeRF/" + DATA
SAVE = "checkpoint/" + DATA + (('_' + NAME) if NAME else '')
# SAVE = 'checkpoint/rocket_coloc_exr_niceRefllessNRF'
MODEL = ARCH + "v1"
# MODEL = 'mlnrfnrf_basev1_1strocket'
CHECKPOINT = 'checkpoint12.pt'  # 'checkpoint_last.pt'
MODEL_PATH = SAVE + '/' + MODEL + '/' + CHECKPOINT

# Rocket
# RENDER_PATH_ARGS = '{\'radius\':1.5,\'h\':3,\'o\':(-0.1,0.05,1.25)}'  # top diagonal view
RENDER_PATH_ARGS = '{\'radius\':4.5,\'h\':0.0,\'o\':(-0.1,0.05,1.25)}'
RENDER_AT_VECTOR = '"(-0.1,0.05,1.25)"'
## Guitar
# RENDER_PATH_ARGS = '{\'radius\':0.8,\'h\':1.0,\'o\':(0,-0.06,0.5)}'
# RENDER_AT_VECTOR = '"(0, -0.06, 0.575)"'
# ## BRDF_Sphere
# RENDER_PATH_ARGS = '{\'radius\':4,\'h\':2,\'o\':(0,0,0)}'
# RENDER_AT_VECTOR = '"(0,0,0)"'

RENDER_PATH_STYLE = 'circle'
RENDER_SPEED = '2'


RAYMARCHING_TOLERANCE = '0.01'
# MODELOVERRIDES = '{"chunk_size":'+CHUNK_SIZE+',"raymarching_tolerance":'+RAYMARCHING_TOLERANCE+'}'

# PREPROCESS = 'none'  # none/mstd/minmax
# MIN_COLOR = '0'  # '-1'
# BG_COLOR = '1.0,1.0,1.0'  # '1.0,1.0,1.0'
# MODELOVERRIDES = '{\'min_color\':'+MIN_COLOR+'}'
MODELOVERRIDES = {}
NUM_WORKERS = ''


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
if RENDER_PATH_LIGHT:
	parameters += '\n--render-path-light'
parameters += '\n--render-num-frames ' + NUM_FRAMES
if TARGETS_PATH:
	parameters += '\n--targets-path ' + TARGETS_PATH
if DRY_RUN:
	parameters += '\n--render-dry-run'
if 'MIN_COLOR' in locals():
	parameters += '\n--min-color ' + MIN_COLOR
if 'MAX_COLOR' in locals():
	parameters += '\n--max-color ' + MAX_COLOR
if 'BG_COLOR' in locals():
	parameters += '\n--transparent-background "' + BG_COLOR + '"'
if 'LIGHT_INTENSITY' in locals():
	parameters += '\n--light-intensity ' + LIGHT_INTENSITY

# parameters += '\n--model-overrides \'{"chunk_size":'+CHUNK_SIZE+',"raymarching_tolerance":0.01}\''
parameters += '\n--render-beam ' + RENDER_BEAM
parameters += '\n--render-save-fps 24'
# parameters += '\n--render-camera-poses ' + DATASET + '/test_traj.txt'
if len(MODELOVERRIDES):
	parameters += '\n--model-overrides ' + MODELOVERRIDES
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



if SAVE_FILE:
	with open('configuration_render.txt', 'w') as f:
		f.write(parameters)

if COPY2CLIPBOARD:
	pyperclip.copy(parameters)

if INJECT_PYCHARM:
	inject_pycharm_config('render', XML_PATH, parameters, NUM_BACKUPS)