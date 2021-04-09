# yeeeey

from types import SimpleNamespace


options = {
	"PATH_MODEL": 'models/trophy.blend',
	"DATASET_NAME": 'trophy_static_png',

	"VOXEL_NUMS": 64,  # 512,  # can still be later overridden using argument 'VOXEL_NUM'
	"VIEWS": 200,  # number of renderings
	"RESOLUTION": 800,  # resolution of resulting renders
	
	"LIGHT_SETUP": 'fixed', # none/fixed/colocated

	"COLOR_DEPTH": 8,  # 16
	"FORMAT": 'PNG',  # 'PNG'/OPEN_EXR'/'HDR' # use 16/32 bit color depth with OPEN_EXR format
	"CYCLES_SAMPLES": 7500,
	"CYCLES_MAX_BOUNCES": 20,

	"CAM_DISTANCE": 1.0,


	"DEBUG": False,

	"DEPTH_SCALE": 1.4,
	"RANDOM_VIEWS": True,  # if False specify "CIRCLE_FIXED_START": (.3,0,0)
	"UPPER_VIEWS": True,
	"RESULTS_PATH": 'rgb',
}
opts = SimpleNamespace(**options)

