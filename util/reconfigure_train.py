import os, os.path as op, sys, time
import pyperclip
import xml.etree.ElementTree as ET
from shutil import copyfile

from reconfigure_utils import inject_pycharm_config

COPY2CLIPBOARD = False  # after running the script the configuration is inserted into clipboard
INJECT_PYCHARM = True
SAVE_FILE = True

DATA = "rocket_random_png"
NAME = "test"  # postfix for dataset name
RES = "64x64"
PIXELS_PER_VIEW = '80'
VIEW_PER_BATCH = '2'  # not sure, but better to be an even divisor of PIXELS_PER_VIEW
SCENE_SCALE = '1.0'

USE_OCTREE = True
USE_CPU = False  # WARNING: does not work on CPU
CHUNK_SIZE = '256'#'256'  # > 1 to save memory to time
LR = '0.001'  # 0.001
VOXEL_NUM = '64'  # '512'  # mutually exclusive with VOXEL_SIZE = 0.27057

# ARCH = "nsvf_base"  # Original NSVF from facebook
# TASK = 'single_object_rendering'

# ARCH = "mlnrf_base"  # Implicit model with ignoring light interaction
# TASK = 'single_object_light_rendering'

# ARCH = "mlnrfiva_base"  # Implicit model with InVoxelApproximation light interaction
# TASK = 'single_object_light_rendering'

ARCH = "mlnrfex_base"  # Explicit model with ignoring light interaction
TASK = 'single_object_light_rendering'

SUFFIX = "v1"
DATASET = "datasets/" + DATA  # "data/Synthetic_NeRF/" + DATA
SAVE = "checkpoint/" + DATA + (('_' + NAME) if NAME else '')
MODEL = ARCH + SUFFIX
#TODO: VOXEL_NUM & VOXEL_SIZE might not work as intended!


REDUCE_STEP_SIZE_AT = '2250,8500,35000'  # '5000,25000,75000'
HALF_VOXEL_SIZE_AT = '5000,25000,50000'  # '5000,25000,75000'
PRUNNING_EVERY_STEPS = '5500'
SAVE_INTERVAL_UPDATES = '500'#'750'  # '100'
TOTAL_NUM_UPDATE = '75000'  # 150000

PREPROCESS = 'none'  # none/mstd/minmax
MIN_COLOR = '-1'  # '-1'
BG_COLOR = '1.0,1.0,1.0'  # '1.0,1.0,1.0'


XML_PATH = '.run/train.run.xml'
NUM_BACKUPS = 10


# create directory if doesn't exist
# if not os.path.exists(SAVE + '/' + MODEL): os.makedirs(SAVE + '/' + MODEL)

# create configuration file
parameters = ''
parameters += DATASET
# if WITH_LIGHT:
# 	parameters += '\n--with-point-light'
# 	parameters += '\n--inputs-to-texture "feat:0:256,ray:4,light:4,lightd:0:1"'
parameters += '\n--user-dir fairnr'
parameters += '\n--task ' + TASK
parameters += '\n--train-views "0..100"'
parameters += '\n--chunk-size '+CHUNK_SIZE
if 'VOXEL_NUM' in locals():
	parameters += '\n--voxel-num ' + locals()['VOXEL_NUM']
elif 'VOXEL_SIZE' in locals():
	parameters += '\n--voxel-size ' + locals()['VOXEL_SIZE']
parameters += '\n--scene-scale ' + SCENE_SCALE
parameters += '\n--view-resolution ' + RES
parameters += '\n--max-sentences 1'
parameters += '\n--view-per-batch ' + VIEW_PER_BATCH
parameters += '\n--pixel-per-view ' + PIXELS_PER_VIEW
parameters += '\n--no-preload'
parameters += '\n--sampling-on-mask 1.0'
parameters += '\n--no-sampling-at-reader'
parameters += '\n--valid-view-resolution ' + RES
parameters += '\n--valid-views "100..200"'
parameters += '\n--valid-view-per-batch 1'
parameters += '\n--transparent-background "' + BG_COLOR + '"'
# parameters += '\n--no-background-loss'
parameters += '\n--background-stop-gradient'
parameters += '\n--arch ' + ARCH
parameters += '\n--initial-boundingbox ' + DATASET + '/bbox.txt'
parameters += '\n--raymarching-stepsize-ratio 0.125'
if USE_OCTREE:
	parameters += '\n--use-octree'
if USE_CPU:
	parameters += '\n--cpu'
parameters += '\n--discrete-regularization'
parameters += '\n--color-weight 128.0'
parameters += '\n--alpha-weight 1.0'
parameters += '\n--min-color ' + MIN_COLOR
parameters += '\n--preprocess ' + PREPROCESS
parameters += '\n--optimizer "adam"'
parameters += '\n--adam-betas "(0.9, 0.999)"'
parameters += '\n--lr-scheduler "polynomial_decay"'
parameters += '\n--total-num-update ' + TOTAL_NUM_UPDATE
parameters += '\n--lr ' + LR
parameters += '\n--clip-norm 0.0'  # 0.01
parameters += '\n--criterion "srn_loss"'
parameters += '\n--num-workers 0'
parameters += '\n--seed 2'
parameters += '\n--save-interval-updates ' + SAVE_INTERVAL_UPDATES
parameters += '\n--max-update 150000'
parameters += '\n--virtual-epoch-steps 5000'
parameters += '\n--save-interval 1'
parameters += '\n--half-voxel-size-at  "' + HALF_VOXEL_SIZE_AT + '"'
parameters += '\n--reduce-step-size-at "' + REDUCE_STEP_SIZE_AT + '"'
parameters += '\n--pruning-every-steps ' + PRUNNING_EVERY_STEPS
# '--rendering-every-steps'
parameters += '\n--keep-interval-updates 5'
parameters += '\n--log-format simple'
parameters += '\n--log-interval 1'
parameters += '\n--tensorboard-logdir ' + SAVE + '/tensorboard/' + MODEL
parameters += '\n--save-dir ' + SAVE + '/' + MODEL

if SAVE_FILE:
	with open('configuration.txt', 'w') as f:
		f.write(parameters)

if COPY2CLIPBOARD:
	pyperclip.copy(parameters)

if INJECT_PYCHARM:
	inject_pycharm_config('train', XML_PATH, parameters, NUM_BACKUPS)