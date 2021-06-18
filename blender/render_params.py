from types import SimpleNamespace
import re

options_train = {
	"START_FROM": 0,

	"PATH_MODEL": 'models/rocket.blend',
	"DATASET_NAME": 'rocket_random_png',
	"OUTPUT_DIR": '%DATASET_NAME%',
	# "PRESET_VIEWS_FOLDER": 'checkpoints/%DATASET_NAME%/mlnrf_base/output_cam',
	"RESULTS_PATH": 'rgb',

	"VOXEL_NUMS": 64,  # 512,  # can still be later overridden using argument 'VOXEL_NUM'
	"VIEWS": 500,  # number of renderings. Ignored, if RANDOM_VIEWS == False
	"RESOLUTION": 1024,  # resolution of resulting renders
	
	"LIGHT_SETUP": 'random', # none/fixed/colocated/random. Ignored, if RANDOM_VIEWS == False

	"COLOR_DEPTH": 8,  # 16
	"FORMAT": 'PNG',  # 'PNG'/OPEN_EXR'/'HDR' # use 16/32 bit color depth with OPEN_EXR format
	"CYCLES_SAMPLES": 5000,#7000,
	"CYCLES_MAX_BOUNCES": 40,#20,

	# "CAM_DISTANCE": 1.0,
	"CAM_HEMISPHERE_ANGLES": [-10, 80],  # in degrees
	"RANDOM_VIEWS": True,  # VIEWS_PATH & LIGHTS_PATH must be specified if RANDOM_VIEWS == False
	# "VIEWS_PATH": '%PRESET_VIEWS_FOLDER%/pose',
	# "LIGHTS_PATH": '%PRESET_VIEWS_FOLDER%/pose_pl',

	"DEBUG": False,

	"DEPTH_SCALE": 1.4
}

options_render = {
	"START_FROM": 0,

	"PATH_MODEL": 'models/rocket.blend',
	"DATASET_NAME": 'rocket_random_png',
	# "MOVING_TYPE": 'light',
	"MOVING_TYPE": 'cam',

	"OUTPUT_DIR": '%DATASET_NAME%_target_%MOVING_TYPE%',
	"PRESET_VIEWS_FOLDER": 'checkpoints/%DATASET_NAME%/mlnrf_base/output_middle_%MOVING_TYPE%',
	"VIEWS_PATH": '%PRESET_VIEWS_FOLDER%/pose',
	"LIGHTS_PATH": '%PRESET_VIEWS_FOLDER%/pose_pl',

	"VOXEL_NUMS": 64,  # 512,  # can still be later overridden using argument 'VOXEL_NUM'
	"RESOLUTION": 256,  # resolution of resulting renders

	"COLOR_DEPTH": 8,  # 16
	"FORMAT": 'PNG',  # 'PNG'/OPEN_EXR'/'HDR' # use 16/32 bit color depth with OPEN_EXR format
	"CYCLES_SAMPLES": 500,#7000,
	"CYCLES_MAX_BOUNCES": 12,#20,

	# "CAM_DISTANCE": 1.0,
	"CAM_HEMISPHERE_ANGLES": [-10, 80],  # in degrees
	"RANDOM_VIEWS": False,  # VIEWS_PATH & LIGHTS_PATH must be specified if RANDOM_VIEWS == False

	"DEBUG": False,
	"DEPTH_SCALE": 1.4,
	"RESULTS_PATH": 'target',
}

# options = options_train; print('\n'.join([''.join(['=']*10)]*3), '>>>>> TRAIN <<<<<');
options = options_render; print('\n'.join([''.join(['=']*10)]*3), '>>>>> RENDER <<<<<');

# Substitute vars
for key in options:
	if not isinstance(options[key], str): continue
	for match in re.finditer('%[A-Z_]+%', options[key]):
		matchKey = match.group()[1:-1]
		if matchKey in options:
			# options[key] = options[key][:match.start()] + options[matchKey] + options[key][match.end():]
			options[key] = options[key].replace(match.group(), options[matchKey])

opts = SimpleNamespace(**options)