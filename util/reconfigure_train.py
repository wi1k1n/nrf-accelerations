import os, os.path as op, sys
import pyperclip
import xml.etree.ElementTree as ET
from shutil import copyfile

COPY2CLIPBOARD = False  # after running the script the configuration is inserted into clipboard
INJECT_PYCHARM = True
SAVE_FILE = True

DATA = "bunny_static_exr"
# DATA = "ttoy_static_exr"
RES = "100x100"
PIXELS_PER_VIEW = '80'  # should be powers of 2 (?)
WITH_LIGHT = False
ARCH = "mlnrf_base"
SUFFIX = "v1"
DATASET = "/home/mazlov/documents/thesis/codes/blender/" + DATA  # "data/Synthetic_NeRF/" + DATA
SAVE = "checkpoint/" + DATA
MODEL = ARCH + SUFFIX
USE_OCTREE = True
CHUNK_SIZE = '128'  # > 1 to save memory to time
LR = '0.001'  # 0.001
VOXEL_NUM = '512'  # '512'  # mutually exclusive with VOXEL_SIZE = 0.27057
#TODO: VOXEL_NUM & VOXEL_SIZE might not work as intended!


HALF_VOXEL_SIZE_AT = '2000,8500,35000'  # '5000,25000,75000'
REDUCE_STEP_SIZE_AT = '2000,8500,35000'  # '5000,25000,75000'
PRUNNING_EVERY_STEPS = '1500'  # '2500'
SAVE_INTERVAL_UPDATES = '100'  # '500'
TOTAL_NUM_UPDATE = '75000'  # 150000

PREPROCESS = 'mstd'  # none/mstd/minmax
MIN_COLOR = '-1'  # '-1'
BG_COLOR = '1.0,1.0,1.0'  # '1.0,1.0,1.0'


XML_PATH = '.run/train.run.xml'

# create directory if doesn't exist
# if not os.path.exists(SAVE + '/' + MODEL): os.makedirs(SAVE + '/' + MODEL)

# create configuration file
parameters = ''
parameters += DATASET
if WITH_LIGHT:
	parameters += '\n--with-point-light'
	parameters += '\n--inputs-to-texture "feat:0:256,ray:4,light:4,lightd:0:1"'
parameters += '\n--user-dir fairnr'
parameters += '\n--task single_object_rendering'
parameters += '\n--train-views "0..100"'
parameters += '\n--chunk-size '+CHUNK_SIZE
if 'VOXEL_NUM' in locals():
	parameters += '\n--voxel-num ' + locals()['VOXEL_NUM']
elif 'VOXEL_SIZE' in locals():
	parameters += '\n--voxel-size ' + locals()['VOXEL_SIZE']
parameters += '\n--view-resolution ' + RES
parameters += '\n--max-sentences 1'
parameters += '\n--view-per-batch 2'
parameters += '\n--pixel-per-view ' + PIXELS_PER_VIEW
parameters += '\n--no-preload'
parameters += '\n--sampling-on-mask 1.0'
parameters += '\n--no-sampling-at-reader'
parameters += '\n--valid-view-resolution ' + RES
parameters += '\n--valid-views "100..200"'
parameters += '\n--valid-view-per-batch 1'
parameters += '\n--transparent-background "' + BG_COLOR + '"'
parameters += '\n--background-stop-gradient'
parameters += '\n--arch ' + ARCH
parameters += '\n--initial-boundingbox ' + DATASET + '/bbox.txt'
parameters += '\n--raymarching-stepsize-ratio 0.125'
if USE_OCTREE:
	parameters += '\n--use-octree'
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
	tree = ET.parse(op.abspath(XML_PATH))
	all_configs = tree.findall('configuration')
	for config in all_configs:
		if not ('name' in config.attrib) or config.get('name') != 'train': continue
		for option in config.findall('option'):
			if not ('name' in option.attrib) or option.get('name') != 'PARAMETERS': continue
			assert ('value' in option.attrib), 'Theres no VALUE attribute in this configuration! Please check!'
			option.set('value', parameters)
			break
		break

	# Create backup
	copyfile(op.abspath(XML_PATH), op.abspath(XML_PATH + '.backup'))

	tree.write(op.abspath(XML_PATH))