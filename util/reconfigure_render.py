import os, os.path as op, sys
from reconfigure_utils import inject_pycharm_config
import pyperclip

DATA = "lego_random_exr"
DATA_MODEL = ''  # data_folder for the model checkpoints; == DATA if empty
NAME = ""  # postfix for dataset name (e.g. u4101)
RENDER_OUTPUT = "random3"  # output if empty
RES = "512x512"
# RENDER_PATH_LIGHT = True  # True - light source is moving, False - camera is moving
# TARGETS_PATH = '/data/mazlov2/Documents/thesis/codes/blender/'\
# 			   + DATA + '_' + NAME + '_target_' + ('light' if RENDER_PATH_LIGHT else 'cam') + '/target'
# TARGETS_PATH = 'datasets/' + DATA + ('_' if NAME else '') + NAME + '_target_' + ('light' if RENDER_PATH_LIGHT else 'cam') + '/target'
TARGETS_PATH = 'datasets/' + DATA + '_random3' + '/target'
# TARGETS_PATH = '/data/mazlov2/Documents/thesis/codes/blender/guitar_coloc_exr_target_cam/target'
# TARGETS_PATH = '/tmp/mazlov/blender/guitar_coloc_exr_target_cam/target'
DRY_RUN = False  # only create camera/light positions and do not evaluate model

CHUNK_SIZE = 2
RENDER_BEAM = '1'  # should be an even divisor of NUM_FRAMES TODO: fix it
NUM_WORKERS = '10'

PREPROCESS = 'log' # none/mstd/minmax/log/nsvf(min_color==-1!)
MIN_COLOR = '0.0'  #
MAX_COLOR = '0.8'  # 0.8 - rocket/guitar/lego/hotdog; 5.0 - sphere; 0.3 - drums; 0.6 - lego-random
GAMMA_CORRECTION = '1.0'  # 2.0 - rocket/guitar/drums; 1.0 - sphere/lego; 1.5 - hotdog
BG_COLOR = '0.0'  # '0.25,0.25,0.25'  # '1.0,1.0,1.0'

TASK = 'single_object_light_rendering'
LAMBERT_ONLY = False
TRACE_NORMAL = False
MODELOVERRIDES = {
	'valid_chunk_size': CHUNK_SIZE
}


# # <!-- Original NSVF from facebook -->
# ARCH = "nsvf_base"
# TASK = 'single_object_rendering'
# # <!/-- Original NSVF from facebook -->

# ARCH = "mlnrfnrf_base"  # Explicit model with NRF (colocated!) light interaction
# MODELOVERRIDES.update({'evaluate_novel_light': True, 'light_intensity': 1000})

ARCH = "mlnrf_base"  # Implicit model with ignoring light interaction
# ARCH = "mlnrfexva_base"  # Explicit model with VoxelApproximation light interaction
# ARCH = "mlnrfexbf_base"  # Explicit model with Brute Force light interaction


# ARCH = "mlnrfiva_base"  # Implicit model with InVoxelApproximation light interaction
# ARCH = "mlnrfex_base"  # Explicit model with ignoring light interaction

DATASET = "datasets/" + DATA  # "data/Synthetic_NeRF/" + DATA
SAVE = "checkpoint/" + (DATA_MODEL if DATA_MODEL else DATA) + (('_' + NAME) if NAME else '')
# SAVE = 'checkpoint/rocket_coloc_exr_niceRefllessNRF'
MODEL = ARCH + "v1"
# MODEL = 'mlnrfnrf_basev1_1strocket'
CHECKPOINT = 'checkpoint_last.pt'  # 'checkpoint_last.pt'
MODEL_PATH = SAVE + '/' + MODEL + '/' + CHECKPOINT

####################################################
#### Hotdog - pl: (1.6 3.0 2.0) -> (a=61.93; h=2.0) camDst=4.0
# RENDER_AT_VECTOR = '"(0, 0, 0)"'
# # === vs_lpan ===
# RENDER_PATH_ARGS = 			{'radius': 3.4, 'h': 2.0, 'o': (0, 0, 0), 't0': 61.93, 'axis': 'z'}
# RENDER_LIGHT_PATH_ARGS = 	{'radius': 3.4, 'h': 2.0, 'o': (0, 0, 0), 't0': 61.93, 'axis': 'z'}
# # === vpan_ls ===
# RENDER_PATH_ARGS = 			{'radius': 3.4, 'h': 2.0, 'o': (0, 0, 0), 't0': 61.93, 'axis': 'z'}
# RENDER_LIGHT_PATH_ARGS = 	{'radius': 3.4, 'h': 2.0, 'o': (0, 0, 0), 't0': 61.93, 'axis': 'z'}

####################################################
#### Guitar - pl: (1.37, -1.22, 1.38) -> (a=41.7; h=1.38; r=1.83) camDst=1.6
# RENDER_AT_VECTOR = '"(0, -0.06, 0.575)"'
# # === vs_lpan ===
# RENDER_PATH_ARGS = 			{'radius': 1.7, 'h': 0.6, 'o': (0, -0.06, 0.5), 't0': -55, 'axis': 'z'}
# RENDER_LIGHT_PATH_ARGS = 	{'radius': 1.83, 'h': 1.38, 'o': (0, -0.06, 0.5), 't0': 41.7, 'axis': 'z'}
# # === vpan_ls ===
# RENDER_PATH_ARGS = 			{'radius': 1.7, 'h': 0.6, 'o': (0, -0.06, 0.5), 't0': -55, 'axis': 'z'}
# RENDER_LIGHT_PATH_ARGS = 	{'radius': 1.83, 'h': 1.38, 'o': (0, -0.06, 0.5), 't0': 41.7, 'axis': 'z'}

# # RENDER_PATH_ARGS = '"{\'radius\':0.8,\'h\':1.0,\'o\':(0,-0.06,0.5),\'t0\':0}"'
# # RENDER_LIGHT_PATH_ARGS = '"{\'radius\':0.8,\'h\':1.0,\'o\':(0,-0.06,0.5),\'t0\':0}"'
# RENDER_PATH_ARGS = '"{\'radius\':1.7,\'h\':0.5,\'o\':(0,-0.06,0.5),\'t0\':-90}"'
# RENDER_LIGHT_PATH_ARGS = '"{\'radius\':1.7,\'h\':0.5,\'o\':(0,-0.06,0.5),\'t0\':-90}"'
####################################################
#### BRDF_Sphere
# RENDER_PATH_ARGS = '{\'radius\':4,\'h\':2,\'o\':(0,0,0)}'
# RENDER_AT_VECTOR = '"(0,0,0)"'

####################################################
#### Lego - pl: (-2.2 2.7 2.7) -> (a=129.17; h=2.7)
RENDER_AT_VECTOR = '"(0, 0, 0)"'
# # === vs_lpan ===
# RENDER_PATH_ARGS = 			{'radius': 3.8, 'h': 2.7, 'o': (0, 0, 0), 't0': 180, 'axis': 'z'}
# RENDER_LIGHT_PATH_ARGS = 	{'radius': 3.8, 'h': 2.7, 'o': (0, 0, 0), 't0': 129.17, 'axis': 'z'}
# # === vcirc_ls ===
# RENDER_PATH_ARGS = 			{'radius': 1.5, 'h': 0, 'o': (-3.8, 0, 2.7-1.5), 't0': 90, 'axis': 'x'}
# RENDER_LIGHT_PATH_ARGS = 	{'radius': 3.8, 'h': 2.7, 'o': (0, 0, 0), 't0': 129.17, 'axis': 'z'}
# === vpan_ls ===
RENDER_PATH_ARGS = 			{'radius': 3.8, 'h': 2.7, 'o': (0, 0, 0), 't0': 180, 'axis': 'z'}
RENDER_LIGHT_PATH_ARGS = 	{'radius': 3.8, 'h': 2.7, 'o': (0, 0, 0), 't0': 129.17, 'axis': 'z'}
# # === vpan_lpan ===
# RENDER_PATH_ARGS = 			{'radius': 3.8, 'h': 2.7, 'o': (0, 0, 0), 't0': 180, 'axis': 'z'}
# RENDER_LIGHT_PATH_ARGS = 	{'radius': 3.8, 'h': 2.7, 'o': (0, 0, 0), 't0': 180, 'axis': 'z'}

NUM_FRAMES = '180' #180
RENDER_PATH_STYLE = 'circle'
RENDER_SPEED = '2'
RENDER_LIGHT_PATH_STYLE = 'circle'
RENDER_LIGHT_SPEED = '0'
FPS = '24'

####################################################
#### Rocket - pl: (-0.98 -2.5 7.2) -> (a=111.4; h=7.2) camDst=5.0
# # RENDER_PATH_ARGS = '{\'radius\':1.5,\'h\':3,\'o\':(-0.1,0.05,1.25)}'  # top diagonal view
# RENDER_PATH_ARGS = '{\'radius\':4.5,\'h\':0.0,\'o\':(-0.1,0.05,1.25)}'
# RENDER_AT_VECTOR = '"(-0.1,0.05,1.25)"'


RAYMARCHING_TOLERANCE = 0.01
MODELOVERRIDES.update({
	'chunk_size': CHUNK_SIZE,
	'raymarching_tolerance': RAYMARCHING_TOLERANCE
})
# MODELOVERRIDES = '"{\'chunk_size\':'+CHUNK_SIZE+',\'raymarching_tolerance\':'+RAYMARCHING_TOLERANCE+'}"'

# PREPROCESS = 'none'  # none/mstd/minmax
# MIN_COLOR = '0'  # '-1'
# BG_COLOR = '1.0,1.0,1.0'  # '1.0,1.0,1.0'
# MODELOVERRIDES = '{\'min_color\':'+MIN_COLOR+'}'
# MODELOVERRIDES = {}




COPY2CLIPBOARD = False
INJECT_PYCHARM = True
SAVE_FILE = True
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
parameters += '\n--render-output ' + SAVE + '/' + ARCH + '/' + RENDER_OUTPUT
parameters += '\n--render-path-style ' + RENDER_PATH_STYLE
parameters += '\n--render-path-args ' + '"{}"'.format(str(RENDER_PATH_ARGS).replace("'", "\'"))
parameters += '\n--render-at-vector ' + RENDER_AT_VECTOR
parameters += '\n--render-angular-speed ' + RENDER_SPEED
# if RENDER_PATH_LIGHT:
# 	parameters += '\n--render-path-light'
parameters += '\n--render-light-path-style ' + RENDER_LIGHT_PATH_STYLE
# parameters += '\n--render-light-path-args ' + RENDER_LIGHT_PATH_ARGS
parameters += '\n--render-light-path-args ' + '"{}"'.format(str(RENDER_LIGHT_PATH_ARGS).replace("'", "\'"))
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
if 'NRF_NOVEL_LIGHT' in locals() and NRF_NOVEL_LIGHT:
	parameters += '\n--evaluate-novel-light'
parameters += '\n--render-beam ' + RENDER_BEAM
parameters += '\n--render-save-fps ' + FPS
# parameters += '\n--render-camera-poses ' + DATASET + '/test_traj.txt'
# if len(MODELOVERRIDES):
# 	parameters += '\n--model-overrides ' + MODELOVERRIDES
if len(MODELOVERRIDES.items()):
	parameters += '\n--model-overrides ' + '"{}"'.format(str(MODELOVERRIDES))
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