import os, os.path as op, sys
from reconfigure_utils import inject_pycharm_config
import pyperclip


COPY2CLIPBOARD = False
INJECT_PYCHARM = True
SAVE_FILE = True

DATA = "lego_static_exr"
NAME = "u4110"  # postfix for dataset name
RENDER_OUTPUT = "test"  # output if empty
RES = "256x256"
# RENDER_PATH_LIGHT = True  # True - light source is moving, False - camera is moving
# TARGETS_PATH = '/data/mazlov2/Documents/thesis/codes/blender/'\
# 			   + DATA + '_' + NAME + '_target_' + ('light' if RENDER_PATH_LIGHT else 'cam') + '/target'
# TARGETS_PATH = 'datasets/' + DATA + ('_' if NAME else '') + NAME + '_target_' + ('light' if RENDER_PATH_LIGHT else 'cam') + '/target'
TARGETS_PATH = 'datasets/' + DATA + ('_' if NAME else '') + 'target_' + ('light' if True else 'cam') + '/target'
# TARGETS_PATH = '/data/mazlov2/Documents/thesis/codes/blender/guitar_coloc_exr_target_cam/target'
# TARGETS_PATH = '/tmp/mazlov/blender/guitar_coloc_exr_target_cam/target'
DRY_RUN = False  # only create camera/light positions and do not evaluate model

CHUNK_SIZE = '2'
RENDER_BEAM = '2'  # should be an even divisor of NUM_FRAMES TODO: fix it
NUM_WORKERS = '8'

PREPROCESS = 'log' # none/mstd/minmax/log/nsvf(min_color==-1!)
MIN_COLOR = '0.0'  #
MAX_COLOR = '0.8'  # 0.8 - rocket/guitar/lego/hotdog; 5.0 - sphere; 0.3 - drums; 0.6 - lego-random
GAMMA_CORRECTION = '1.0'  # 2.0 - rocket/guitar/drums; 1.0 - sphere/lego; 1.5 - hotdog
BG_COLOR = '0.0'  # '0.25,0.25,0.25'  # '1.0,1.0,1.0'

TASK = 'single_object_light_rendering'
LAMBERT_ONLY = False
TRACE_NORMAL = False

# WITH_LIGHT = True
# ARCH = "mlnrf_base"
# TASK = 'single_object_light_rendering'
# ARCH = "nsvf_base"
# TASK = 'single_object_rendering'


# # <!-- Original NSVF from facebook -->
# ARCH = "nsvf_base"
# TASK = 'single_object_rendering'
# # <!/-- Original NSVF from facebook -->

# <!-- Implicit model with ignoring light interaction -->
ARCH = "mlnrf_base"
# <!/-- Implicit model with ignoring light interaction -->

# # <!-- Implicit model with InVoxelApproximation light interaction -->
# ARCH = "mlnrfiva_base"
# # <!/-- Implicit model with InVoxelApproximation light interaction -->

# # <!-- Explicit model with ignoring light interaction -->
# ARCH = "mlnrfex_base"
# TRACE_NORMAL = True
# LAMBERT_ONLY = False
# # <!/-- Explicit model with ignoring light interaction -->

# # <!-- Explicit model with NRF (colocated!) light interaction -->
# ARCH = "mlnrfnrf_base"
# # <!/-- Explicit model with ignoring light interaction -->

# # <!-- Explicit model with VoxelApproximation light interaction -->
# ARCH = "mlnrfexva_base"
# # <!/-- Explicit model with ignoring light interaction -->

# # <!-- Explicit model with Brute Force light interaction -->
# ARCH = "mlnrfexbf_base"
# # <!/-- Explicit model with Brute Force light interaction -->


DATASET = "datasets/" + DATA  # "data/Synthetic_NeRF/" + DATA
SAVE = "checkpoint/" + DATA + (('_' + NAME) if NAME else '')
# SAVE = 'checkpoint/rocket_coloc_exr_niceRefllessNRF'
MODEL = ARCH + "v1"
# MODEL = 'mlnrfnrf_basev1_1strocket'
CHECKPOINT = 'checkpoint_last.pt'  # 'checkpoint_last.pt'
MODEL_PATH = SAVE + '/' + MODEL + '/' + CHECKPOINT

####################################################
#### Rocket
# # RENDER_PATH_ARGS = '{\'radius\':1.5,\'h\':3,\'o\':(-0.1,0.05,1.25)}'  # top diagonal view
# RENDER_PATH_ARGS = '{\'radius\':4.5,\'h\':0.0,\'o\':(-0.1,0.05,1.25)}'
# RENDER_AT_VECTOR = '"(-0.1,0.05,1.25)"'
####################################################
#### Guitar
# RENDER_AT_VECTOR = '"(0, -0.06, 0.575)"'
# # RENDER_PATH_ARGS = '"{\'radius\':0.8,\'h\':1.0,\'o\':(0,-0.06,0.5),\'t0\':0}"'
# # RENDER_LIGHT_PATH_ARGS = '"{\'radius\':0.8,\'h\':1.0,\'o\':(0,-0.06,0.5),\'t0\':0}"'
# RENDER_PATH_ARGS = '"{\'radius\':1.7,\'h\':0.5,\'o\':(0,-0.06,0.5),\'t0\':-90}"'
# RENDER_LIGHT_PATH_ARGS = '"{\'radius\':1.7,\'h\':0.5,\'o\':(0,-0.06,0.5),\'t0\':-90}"'
####################################################
#### BRDF_Sphere
# RENDER_PATH_ARGS = '{\'radius\':4,\'h\':2,\'o\':(0,0,0)}'
# RENDER_AT_VECTOR = '"(0,0,0)"'
####################################################
#### Lego  (pl: (-2.2 2.7 2.7) -> (a=129.17; h=2.7)
RENDER_AT_VECTOR = '"(0, 0, 0)"'
RENDER_PATH_ARGS = '"{\'radius\':3.8,\'h\':2.7,\'o\':(0,0,0),\'t0\':129.17}"'
RENDER_LIGHT_PATH_ARGS = '"{\'radius\':3.8,\'h\':2.7,\'o\':(0,0,0),\'t0\':129.17}"'
# RENDER_PATH_ARGS = '"{\'radius\':3.8,\'h\':2.0,\'o\':(0,0,0),\'t0\':0}"'
# RENDER_LIGHT_PATH_ARGS = '"{\'radius\':3.8,\'h\':2.0,\'o\':(0,0,0),\'t0\':0}"'
# # !!Static Light!!
# RENDER_PATH_ARGS = '"{\'radius\':3.8,\'h\':2.7,\'o\':(0,0,0),\'t0\':129.17}"'
# RENDER_LIGHT_PATH_ARGS = '"{\'radius\':3.8,\'h\':2.7,\'o\':(0,0,0),\'t0\':129.17}"'

NUM_FRAMES = '12' #180
RENDER_PATH_STYLE = 'circle'
RENDER_SPEED = '0'
RENDER_LIGHT_PATH_STYLE = 'circle'
RENDER_LIGHT_SPEED = '30'


RAYMARCHING_TOLERANCE = '0.01'
MODELOVERRIDES = '"{\'chunk_size\':'+CHUNK_SIZE+',\'raymarching_tolerance\':'+RAYMARCHING_TOLERANCE+'}"'

# PREPROCESS = 'none'  # none/mstd/minmax
# MIN_COLOR = '0'  # '-1'
# BG_COLOR = '1.0,1.0,1.0'  # '1.0,1.0,1.0'
# MODELOVERRIDES = '{\'min_color\':'+MIN_COLOR+'}'
# MODELOVERRIDES = {}


XML_PATH = '.run/render.run.xml'
NUM_BACKUPS = 10




# create configuration file
parameters = ''
parameters += DATASET
if DRY_RUN:
	parameters += '\n--render-dry-run'
parameters += '\n--path ' + MODEL_PATH
parameters += '\n--task ' + TASK
if not RENDER_OUTPUT: RENDER_OUTPUT = "output"
parameters += '\n--render-output ' + SAVE + '/' + ARCH + '/' + RENDER_OUTPUT + ('_light' if True else '_cam')
parameters += '\n--render-path-style ' + RENDER_PATH_STYLE
parameters += '\n--render-path-args ' + RENDER_PATH_ARGS
parameters += '\n--render-at-vector ' + RENDER_AT_VECTOR
parameters += '\n--render-angular-speed ' + RENDER_SPEED
# if RENDER_PATH_LIGHT:
# 	parameters += '\n--render-path-light'
parameters += '\n--render-light-path-style ' + RENDER_LIGHT_PATH_STYLE
parameters += '\n--render-light-path-args ' + RENDER_LIGHT_PATH_ARGS
parameters += '\n--render-light-angular-speed ' + RENDER_LIGHT_SPEED
parameters += '\n--render-num-frames ' + NUM_FRAMES
if TARGETS_PATH:
	parameters += '\n--targets-path ' + TARGETS_PATH
if 'MIN_COLOR' in locals():
	parameters += '\n--min-color ' + MIN_COLOR
if 'MAX_COLOR' in locals():
	parameters += '\n--max-color ' + MAX_COLOR
if 'GAMMA_CORRECTION' in locals():
	parameters += '\n--gamma-correction ' + GAMMA_CORRECTION
if 'BG_COLOR' in locals():
	parameters += '\n--transparent-background "' + BG_COLOR + '"'
if 'LIGHT_INTENSITY' in locals():
	parameters += '\n--light-intensity ' + LIGHT_INTENSITY
if 'PREPROCESS' in locals():
	parameters += '\n--preprocess ' + PREPROCESS

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
		# f.write(parameters)
		f.write(parameters.replace('\n', ' '))

if COPY2CLIPBOARD:
	pyperclip.copy(parameters)

if INJECT_PYCHARM:
	inject_pycharm_config('render', XML_PATH, parameters, NUM_BACKUPS)