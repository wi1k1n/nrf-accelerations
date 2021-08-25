import os, os.path as op, sys, time
import pyperclip
import xml.etree.ElementTree as ET
from shutil import copyfile

from reconfigure_utils import inject_pycharm_config

COPY2CLIPBOARD = False  # after running the script the configuration is inserted into clipboard
INJECT_PYCHARM = True
SAVE_FILE = True

DATA = "rocket_static_exr"
NAME = ""  # postfix for dataset name
RES = "512x512"
PIXELS_PER_VIEW = '80'
VIEW_PER_BATCH = '2'  # not sure, but better to be an even divisor of PIXELS_PER_VIEW

USE_OCTREE = True
CHUNK_SIZE = '2'  #'256'  # > 1 to save memory to time
LR = '0.0001'  # 0.001
VOXEL_NUM = '64'  # '512'  # mutually exclusive with VOXEL_SIZE = 0.27057

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
TRAIN_VIEWS = '0..160'  # '0..100'
VALID_VIEWS = '160..200'  # '100..200
NUM_WORKERS = '8'  # '0'

HDRFLIP = True
PREPROCESS = 'log'  # none/mstd/minmax/log/nsvf(min_color==-1!)
MIN_COLOR = '0.0'  #
MAX_COLOR = '0.8'  # 0.8 - rocket/guitar/lego/hotdog; 5.0 - sphere; 0.3 - drums; 0.6 - lego-random
GAMMA_CORRECTION = '2.0'  # 2.0 - rocket/guitar/drums; 1.0 - sphere/lego; 1.5 - hotdog
BG_COLOR = '0.0'  # '0.25,0.25,0.25'  # '1.0,1.0,1.0'
SIGMA_NOISE = True
# SIGMA_NOISE_LIGHT = False  # not implemented yet


TRACE_NORMAL = False
LAMBERT_ONLY = False
TASK = 'single_object_light_rendering'

# <!-- Original NSVF from facebook -->
ARCH = "nsvf_base"
TASK = 'single_object_rendering'
# <!/-- Original NSVF from facebook -->

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

# # <!-- Explicit model with VoxelApproximation light interaction -->
# ARCH = "mlnrfexva_base"
# PREDICT_L = True
# VOXEL_SIGMA = 0.5
# # LIGHT_INTENSITY = '1000.0'  # sphere_exr -> 1k Watt
# LIGHT_INTENSITY = '500.0'  # 500 excol; rocket_exr -> 5k Watt
# # LIGHT_INTENSITY = '350.0'  # tablelamp_exr -> 0.5k Watt
# # LIGHT_INTENSITY = '300.0'  # guitar_exr -> 0.5k Watt
# # LIGHT_INTENSITY = '20.0'  # 20 exbf/exva; lego -> 0.7k Watt
# # LIGHT_INTENSITY = '500.0'  # hotdog -> 0.7k Watt
# TEXTURE_LAYERS = '5'
# # <!/-- Explicit model with VoxelApproximation light interaction -->

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
#TODO: VOXEL_NUM & VOXEL_SIZE might not work as intended!


# USE_CPU = False  # WARNING: does not work on CPU
# SCENE_SCALE = '1.0'
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
# parameters += '\n--background-stop-gradient'
parameters += '\n--task ' + TASK
parameters += '\n--train-views "' + TRAIN_VIEWS + '"'
parameters += '\n--chunk-size '+CHUNK_SIZE
parameters += '\n--valid-chunk-size '+CHUNK_SIZE
if 'VOXEL_NUM' in locals():
	parameters += '\n--voxel-num ' + locals()['VOXEL_NUM']
elif 'VOXEL_SIZE' in locals():
	parameters += '\n--voxel-size ' + locals()['VOXEL_SIZE']
if 'TRACE_NORMAL' in locals() and TRACE_NORMAL:
	parameters += '\n--trace-normal'
if 'LAMBERT_ONLY' in locals() and LAMBERT_ONLY:
	parameters += '\n--lambert-only'
if 'PREDICT_L' in locals() and PREDICT_L:
	parameters += '\n--predict-l'
if 'VOXEL_SIGMA' in locals():
	parameters += '\n--voxel-sigma ' + str(VOXEL_SIGMA)
if 'COMPOSITE_R' in locals() and COMPOSITE_R:
	parameters += '\n--composite-r'
if 'GAMMA_CORRECTION' in locals():
	parameters += '\n--gamma-correction ' + GAMMA_CORRECTION
if 'LIGHT_INTENSITY' in locals():
	parameters += '\n--light-intensity ' + LIGHT_INTENSITY
if 'SIGMA_NOISE' in locals() and SIGMA_NOISE:
	parameters += '\n--discrete-regularization'
if 'SIGMA_NOISE_LIGHT' in locals() and SIGMA_NOISE_LIGHT:
	parameters += '\n--discrete-regularization-light'
# parameters += '\n--scene-scale ' + SCENE_SCALE
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
parameters += '\n--initial-boundingbox ' + DATASET + '/bbox.txt'
parameters += '\n--raymarching-stepsize-ratio 0.125'
if USE_OCTREE:
	parameters += '\n--use-octree'
if 'TEXTURE_LAYERS' in locals():
	parameters += '\n--texture-layers ' + TEXTURE_LAYERS
# if USE_CPU:
# 	parameters += '\n--cpu'
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
parameters += '\n--lr ' + LR
parameters += '\n--clip-norm 0.0'  # 0.01
parameters += '\n--criterion "srn_loss"'
parameters += '\n--num-workers ' + NUM_WORKERS
parameters += '\n--seed 2'
parameters += '\n--save-interval-updates ' + SAVE_INTERVAL_UPDATES
parameters += '\n--max-update 150000'
parameters += '\n--virtual-epoch-steps 5000'
parameters += '\n--save-interval 1'
parameters += '\n--half-voxel-size-at  "' + HALF_VOXEL_SIZE_AT + '"'
parameters += '\n--reduce-step-size-at "' + REDUCE_STEP_SIZE_AT + '"'
parameters += '\n--pruning-every-steps ' + PRUNNING_EVERY_STEPS
if 'PRUNNING_TH' in locals():
	parameters += '\n--pruning-th ' + PRUNNING_TH
# '--rendering-every-steps'
parameters += '\n--keep-interval-updates 5'
parameters += '\n--log-format simple'
parameters += '\n--log-interval 1'
parameters += '\n--tensorboard-logdir ' + SAVE + '/tensorboard/' + MODEL
parameters += '\n--save-dir ' + SAVE + '/' + MODEL

if SAVE_FILE:
	with open('configuration.txt', 'w') as f:
		# f.write(parameters)
		f.write(parameters.replace('\n', ' '))

if COPY2CLIPBOARD:
	pyperclip.copy(parameters)

if INJECT_PYCHARM:
	inject_pycharm_config('train', XML_PATH, parameters, NUM_BACKUPS)