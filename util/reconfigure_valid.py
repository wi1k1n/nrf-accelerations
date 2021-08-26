import os, os.path as op, sys, time
import pyperclip
import xml.etree.ElementTree as ET
from shutil import copyfile

from reconfigure_utils import inject_pycharm_config

COPY2CLIPBOARD = False  # after running the script the configuration is inserted into clipboard
INJECT_PYCHARM = True
SAVE_FILE = True

DATA = "lego_random_exr"
NAME = "u4108"  # postfix for dataset name
RES = "256x256"
PIXELS_PER_VIEW = '80'
VIEW_PER_BATCH = '2'  # not sure, but better to be an even divisor of PIXELS_PER_VIEW

USE_OCTREE = True
LR = '0.0001'  # 0.001

COLOR_WEIGHT = '1.0'  #'256.0'
ALPHA_WEIGHT = '1e-3'  #'1e-3'


REDUCE_STEP_SIZE_AT = '5000,25000,50000'
HALF_VOXEL_SIZE_AT = '5000,25000,50000'
PRUNNING_EVERY_STEPS = '5000'
# REDUCE_STEP_SIZE_AT = '10000,50000,100000'  # '5000,25000,75000'
# HALF_VOXEL_SIZE_AT = '10000,50000,100000'  # '5000,25000,75000'
# PRUNNING_EVERY_STEPS = '10000'

PRUNNING_TH = '0.5'  # '0.5'
SAVE_INTERVAL_UPDATES = '1000'#'750'  # '100'
TOTAL_NUM_UPDATE = '150000'  # 150000
TRAIN_VIEWS = '0..475'  # '0..100'
VALID_VIEWS = '475..200'  # '100..200
NUM_WORKERS = '8'  # '0'

HDRFLIP = True
PREPROCESS = 'log'  # none/mstd/minmax/log/nsvf(min_color==-1!)
MIN_COLOR = '0.0'  #
MAX_COLOR = '0.8'  # 0.8 - rocket/guitar/lego/hotdog; 5.0 - sphere; 0.3 - drums; 0.6 - lego-random
GAMMA_CORRECTION = '1.0'  # 2.0 - rocket/guitar/drums; 1.0 - sphere/lego; 1.5 - hotdog
BG_COLOR = '0.0'  # '0.25,0.25,0.25'  # '1.0,1.0,1.0'
SIGMA_NOISE = True
# SIGMA_NOISE_LIGHT = False  # not implemented yet


TRACE_NORMAL = False
LAMBERT_ONLY = False
# TASK = 'single_object_light_rendering'

# # <!-- Original NSVF from facebook -->
# ARCH = "nsvf_base"
# TASK = 'single_object_rendering'
# # <!/-- Original NSVF from facebook -->

# # <!-- Implicit model with ignoring light interaction -->
# ARCH = "mlnrf_base"
# # <!/-- Implicit model with ignoring light interaction -->

# # <!-- Implicit model with InVoxelApproximation light interaction -->
# ARCH = "mlnrfiva_base"
# VOXEL_SIGMA = 0.8
# # <!/-- Implicit model with InVoxelApproximation light interaction -->

# # <!-- Explicit model with ignoring light interaction -->
# ARCH = "mlnrfex_base"
# TRACE_NORMAL = True
# LAMBERT_ONLY = False
# TEXTURE_LAYERS = '4'
# LIGHT_INTENSITY = '1000.0'
# # <!/-- Explicit model with ignoring light interaction -->

# # <!-- Explicit model with NRF (colocated!) light interaction -->
# ARCH = "mlnrfnrf_base"
# PREDICT_L = True
# # LIGHT_INTENSITY = '1000.0'  # sphere_exr -> 1k Watt
# # LIGHT_INTENSITY = '500.0'  # rocket_exr -> 5k Watt
# # LIGHT_INTENSITY = '300.0'  # guitar_exr -> 0.5k Watt
# # LIGHT_INTENSITY = '500.0'#'300.0'  # lego -> 0.7k Watt
# # LIGHT_INTENSITY = '1000.0'  # drums -> 1k Watt
# LIGHT_INTENSITY = '500.0'  # hotdog -> 0.7k Watt
# TEXTURE_LAYERS = '5'
# # <!/-- Explicit model with NRF (colocated!) light interaction -->

# <!-- Explicit model with VoxelApproximation light interaction -->
ARCH = "mlnrfexva_base"
# <!/-- Explicit model with VoxelApproximation light interaction -->

# # <!-- Explicit model with Brute Force light interaction -->
# ARCH = "mlnrfexbf_base"
# PREDICT_L = True
# # LIGHT_INTENSITY = '1000.0'  # sphere_exr -> 1k Watt
# # LIGHT_INTENSITY = '500.0'  # rocket_exr -> 5k Watt
# # LIGHT_INTENSITY = '350.0'  # tablelamp_exr -> 0.5k Watt
# # LIGHT_INTENSITY = '200.0'  # guitar_exr -> 0.5k Watt
# LIGHT_INTENSITY = '20.0'  # lego -> 0.7k Watt
# # LIGHT_INTENSITY = '300.0'  # hotdog -> 0.7k Watt
# TEXTURE_LAYERS = '5'
# # <!/-- Explicit model with Brute Force light interaction -->



LPIPS = True

SUFFIX = "v1"
DATASET = "datasets/" + DATA  # "data/Synthetic_NeRF/" + DATA
SAVE = "checkpoint/" + DATA + (('_' + NAME) if NAME else '')
# SAVE = "checkpoint/rocket_random_exr_test4"
MODEL = ARCH + SUFFIX
CHECKPOINT = 'checkpoint_last.pt'  # 'checkpoint_last.pt'
MODEL_PATH = SAVE + '/' + MODEL + '/' + CHECKPOINT

#TODO: VOXEL_NUM & VOXEL_SIZE might not work as intended!


# USE_CPU = False  # WARNING: does not work on CPU
# SCENE_SCALE = '1.0'
NUM_BACKUPS = 10


# create configuration file
parameters = ''
parameters += DATASET
parameters += '\n--path ' + MODEL_PATH
# parameters += '\n--task ' + TASK
parameters += '\n--train-views "' + TRAIN_VIEWS + '"'
if 'MIN_COLOR' in locals():
	parameters += '\n--min-color ' + MIN_COLOR
if 'MAX_COLOR' in locals():
	parameters += '\n--max-color ' + MAX_COLOR
if 'GAMMA_CORRECTION' in locals():
	parameters += '\n--gamma-correction ' + GAMMA_CORRECTION
if 'BG_COLOR' in locals():
	parameters += '\n--transparent-background "' + BG_COLOR + '"'
if 'PREPROCESS' in locals():
	parameters += '\n--preprocess ' + PREPROCESS
parameters += '\n--view-resolution ' + RES
parameters += '\n--max-sentences 1'
parameters += '\n--view-per-batch ' + VIEW_PER_BATCH
parameters += '\n--pixel-per-view ' + PIXELS_PER_VIEW
parameters += '\n--no-preload'
parameters += '\n--sampling-on-mask 1.0'
parameters += '\n--no-sampling-at-reader'
parameters += '\n--valid-view-resolution ' + RES
parameters += '\n--valid-views "' + VALID_VIEWS + '"'
parameters += '\n--valid-view-per-batch 1'
if 'HDRFLIP' in locals() and HDRFLIP:
	parameters += '\n--eval-hdrflip'
if 'LPIPS' in locals() and LPIPS:
	parameters += '\n--eval-lpips'
parameters += '\n--transparent-background "' + BG_COLOR + '"'
# parameters += '\n--no-background-loss'
parameters += '\n--background-stop-gradient'
parameters += '\n--arch ' + ARCH
parameters += '\n--color-weight ' + COLOR_WEIGHT
parameters += '\n--alpha-weight ' + ALPHA_WEIGHT
parameters += '\n--min-color ' + MIN_COLOR
parameters += '\n--max-color ' + MAX_COLOR
if 'PREPROCESS' in locals():
	parameters += '\n--preprocess ' + PREPROCESS
parameters += '\n--optimizer "adam"'
parameters += '\n--adam-betas "(0.9, 0.999)"'
parameters += '\n--lr-scheduler "polynomial_decay"'
parameters += '\n--total-num-update ' + TOTAL_NUM_UPDATE
parameters += '\n--end-learning-rate ' + str(float(LR) * 1e-2)
# parameters += '\n--criterion "srn_loss"'
if NUM_WORKERS:
	parameters += '\n--num-workers ' + NUM_WORKERS
parameters += '\n--seed 2'
parameters += '\n--log-format simple'
parameters += '\n--log-interval 1'

if SAVE_FILE:
	with open('configuration_valid.txt', 'w') as f:
		# f.write(parameters)
		f.write(parameters.replace('\n', ' '))

if COPY2CLIPBOARD:
	pyperclip.copy(parameters)

if INJECT_PYCHARM:
	inject_pycharm_config('valid', '.run/valid.run.xml', parameters, NUM_BACKUPS)