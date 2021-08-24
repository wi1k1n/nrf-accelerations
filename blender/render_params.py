from types import SimpleNamespace
import re

options_train = {
	"START_FROM": 0,

	"PATH_MODEL": 'models/hotdog.blend',
	"DATASET_NAME": 'hotdog_coloc_exr',
	"OUTPUT_DIR": '%DATASET_NAME%',
	# "PRESET_VIEWS_FOLDER": 'checkpoints/%DATASET_NAME%/mlnrf_base/output_cam',
	"RESULTS_PATH": 'rgb',

	"VOXEL_NUMS": 64,  # 512,  # can still be later overridden using argument 'VOXEL_NUM'
	"VIEWS": 200,  # number of renderings. Ignored, if RANDOM_VIEWS == False
	"RESOLUTION": 512,  # resolution of resulting renders
	
	"LIGHT_SETUP": 'colocated', # none/fixed/colocated/random. Ignored, if RANDOM_VIEWS == False
	"LIGHT_COS_CONSTRAIN": 120, # None/(0 ~ 180). Max angle between cam and light. Ignored, if RANDOM_VIEWS == False
	# "LIGHT_COS_CONSTRAIN_RND": False,  	# False/(mean, std) (e.g. (0, 3*light_cos_constr). 
	# 									# If constain is sharp or stochastic
	# 									# DOESNT WORK!

	"COLOR_DEPTH": 16,  # 8
	"FORMAT": 'OPEN_EXR',  # 'PNG'/OPEN_EXR'/'HDR' # use 16/32 bit color depth with OPEN_EXR format
	"CYCLES_SAMPLES": 1000,#100,
	"CYCLES_MAX_BOUNCES": 15,#4,

	# "CAM_DISTANCE": 1.0,
	"CAM_HEMISPHERE_ANGLES": [-10, 80],  # in degrees
	"RANDOM_VIEWS": True,  # VIEWS_PATH & LIGHTS_PATH must be specified if RANDOM_VIEWS == False
	# "VIEWS_PATH": '%PRESET_VIEWS_FOLDER%/pose',
	# "LIGHTS_PATH": '%PRESET_VIEWS_FOLDER%/pose_pl',

	"DEBUG": False,

	"DEPTH_SCALE": 1.4,

	"PERCENTILE_MIN": 0.5,
	"PERCENTILE_MAX": 99.5,
}

options_render = {
	"START_FROM": 0,

	"PATH_MODEL": 'models/guitar.blend',
	"DATASET_NAME": 'guitar_coloc_exr',
	# "MOVING_TYPE": 'light',
	"MOVING_TYPE": 'cam',
	"RESOLUTION": 256,  # resolution of resulting renders

	"ARCH": 'mlnrfnrf_base',  # nsvf_base/mlnrf_base/mlnrfiva_base/mlnrfex_base/mlnrfnrf_base/mlnrfexbf_base
	"POOLS": '../pool/u4106/checkpoint/', # ../pool/u4101/checkpoint/

	"OUTPUT_DIR": '%DATASET_NAME%_target_%MOVING_TYPE%',
	"PRESET_VIEWS_FOLDER": 'checkpoints/%POOLS%%DATASET_NAME%/%ARCH%/output_%MOVING_TYPE%',
	"VIEWS_PATH": '%PRESET_VIEWS_FOLDER%/pose',
	"LIGHTS_PATH": '%PRESET_VIEWS_FOLDER%/pose_pl',

	"VOXEL_NUMS": 64,  # 512,  # can still be later overridden using argument 'VOXEL_NUM'

	"COLOR_DEPTH": 16,
	"FORMAT": 'OPEN_EXR',
	"CYCLES_SAMPLES": 1000,#7000,
	"CYCLES_MAX_BOUNCES": 20,#20,

	# "CAM_DISTANCE": 1.0,
	"CAM_HEMISPHERE_ANGLES": [-10, 80],  # in degrees
	"RANDOM_VIEWS": False,  # VIEWS_PATH & LIGHTS_PATH must be specified if RANDOM_VIEWS == False

	"DEBUG": False,
	"DEPTH_SCALE": 1.4,
	"RESULTS_PATH": 'target',

	"PERCENTILE_MIN": 0.5,
	"PERCENTILE_MAX": 99.5,
}

options = options_train; print('\n'.join([''.join(['=']*10)]*3), '>>>>> TRAIN <<<<<');
# options = options_render; print('\n'.join([''.join(['=']*10)]*3), '>>>>> RENDER <<<<<');

# Substitute vars
for key in options:
	if not isinstance(options[key], str): continue
	for match in re.finditer('%[A-Z_]+%', options[key]):
		matchKey = match.group()[1:-1]
		if matchKey in options:
			# options[key] = options[key][:match.start()] + options[matchKey] + options[key][match.end():]
			options[key] = options[key].replace(match.group(), options[matchKey])

opts = SimpleNamespace(**options)
