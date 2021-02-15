import os, os.path as op, sys
import pyperclip

COPY2CLIPBOARD = True  # after running the script the configuration is inserted into clipboard

DATA = "bunny_static_exr"
# DATA = "donut_static_png"
RES = "100x100"
PIXELS_PER_VIEW = '32'  # should be powers of 2 (?)
WITH_LIGHT = False
ARCH = "mlnrf_base"
SUFFIX = "v1"
# DATASET = "data/Synthetic_NeRF/" + DATA
DATASET = "/home/mazlov/documents/thesis/codes/blender/" + DATA
SAVE = "checkpoint/" + DATA
MODEL = ARCH + SUFFIX
USE_OCTREE = True
CHUNK_SIZE = '1'  # > 1 to save memory to time
LR = '0.001'  # 0.001

# create directory if doesn't exist
# if not os.path.exists(SAVE + '/' + MODEL): os.makedirs(SAVE + '/' + MODEL)

# create configuration file
with open('configuration.txt', 'w') as f:
	f.write(DATASET)
	if WITH_LIGHT:
		f.write('\n--with-point-light')
		f.write('\n--inputs-to-texture "feat:0:256,ray:4,light:4,lightd:0:1"')
	f.write('\n--user-dir fairnr')
	f.write('\n--task single_object_rendering')
	f.write('\n--train-views "0..100"')
	f.write('\n--chunk-size '+CHUNK_SIZE)
	f.write('\n--view-resolution '+RES)
	f.write('\n--max-sentences 1')
	f.write('\n--view-per-batch 2')
	f.write('\n--pixel-per-view '+PIXELS_PER_VIEW)
	f.write('\n--no-preload')
	f.write('\n--sampling-on-mask 1.0')
	f.write('\n--no-sampling-at-reader')
	f.write('\n--valid-view-resolution '+RES)
	f.write('\n--valid-views "100..200"')
	f.write('\n--valid-view-per-batch 1')
	f.write('\n--transparent-background "1.0,1.0,1.0"')
	f.write('\n--background-stop-gradient')
	f.write('\n--arch '+ARCH)
	f.write('\n--initial-boundingbox '+DATASET+'/bbox.txt')
	f.write('\n--raymarching-stepsize-ratio 0.125')
	if USE_OCTREE: f.write('\n--use-octree')
	f.write('\n--discrete-regularization')
	f.write('\n--color-weight 128.0')
	f.write('\n--alpha-weight 1.0')
	f.write('\n--optimizer "adam"')
	f.write('\n--adam-betas "(0.9, 0.999)"')
	f.write('\n--lr-scheduler "polynomial_decay"')
	f.write('\n--total-num-update 150000')
	f.write('\n--lr '+LR)
	f.write('\n--clip-norm 0.0')
	# f.write('\n--clip-norm 0.01')
	f.write('\n--criterion "srn_loss"')
	f.write('\n--num-workers 0')
	f.write('\n--seed 2')
	f.write('\n--save-interval-updates 500 --max-update 150000')
	f.write('\n--virtual-epoch-steps 5000 --save-interval 1')
	f.write('\n--half-voxel-size-at  "5000,25000,75000"')
	f.write('\n--reduce-step-size-at "5000,25000,75000"')
	f.write('\n--pruning-every-steps 2500')
	# '--rendering-every-steps'
	f.write('\n--keep-interval-updates 5')
	f.write('\n--log-format simple --log-interval 1')
	f.write('\n--tensorboard-logdir '+SAVE+'/tensorboard/'+MODEL)
	f.write('\n--save-dir '+SAVE+'/'+MODEL)

if COPY2CLIPBOARD:
	with open('configuration.txt', 'r') as f:
		pyperclip.copy(f.read())