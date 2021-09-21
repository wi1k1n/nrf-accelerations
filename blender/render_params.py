from types import SimpleNamespace
import re

options_render = {
	"START_FROM": 0,

	"PATH_MODEL": 'models/hotdog.blend',
	"DATASET_NAME": 'hotdog_random_exr',
	"DATAMODEL_NAME": '',  # dataset used for training; == %DATASET_NAME% if empty
	"RESOLUTION": 512,  # resolution of resulting renders

	"ARCH": 'mlnrf_base',  # nsvf_base/mlnrf_base/mlnrfiva_base/mlnrfex_base/mlnrfnrf_base/mlnrfexbf_base/mlnrfexva_base
	"RENDERING_NAME": 'random3',
	# "POOLS": '',
	"POOLS": '../pool/u4109/checkpoint/',

	"COLOR_DEPTH": 16,
	"FORMAT": 'OPEN_EXR',
	"CYCLES_SAMPLES": 500,#7000,
	"CYCLES_MAX_BOUNCES": 20,#20,

	"OUTPUT_DIR": '%DATASET_NAME%_random3true',
	"PRESET_VIEWS_FOLDER": 'checkpoints/%POOLS%%DATASET_NAME%/%ARCH%/%RENDERING_NAME%',
	# "PRESET_VIEWS_FOLDER": 'checkpoints/%POOLS%lego_coloc_exr/%ARCH%/%RENDERING_NAME%',
	"VIEWS_PATH": '%PRESET_VIEWS_FOLDER%/pose',
	"LIGHTS_PATH": '%PRESET_VIEWS_FOLDER%/pose_pl',


	"VOXEL_NUMS": 64,  # 512,  # can still be later overridden using argument 'VOXEL_NUM'
	# "CAM_DISTANCE": 1.0,
	"CAM_HEMISPHERE_ANGLES": [-10, 80],  # in degrees
	"RANDOM_VIEWS": False,  # VIEWS_PATH & LIGHTS_PATH must be specified if RANDOM_VIEWS == False

	"DEBUG": False,
	"DEPTH_SCALE": 1.4,
	"RESULTS_PATH": 'target',

	"PERCENTILE_MIN": 0.5,
	"PERCENTILE_MAX": 99.5,
}
if options_render['DATAMODEL_NAME']:
	options_render['PRESET_VIEWS_FOLDER'] = options_render['PRESET_VIEWS_FOLDER'].replace('%DATASET_NAME%', options_render['DATAMODEL_NAME'])



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
