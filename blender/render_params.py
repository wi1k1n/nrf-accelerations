# yeeeey

from types import SimpleNamespace


options = {
	"PATH_MODEL": 'models/rocket.blend',
	"DATASET_NAME": 'rocket_static_png',

	"VOXEL_NUMS": 64,  # 512,  # can still be later overridden using argument 'VOXEL_NUM'
	"VIEWS": 200,  # number of renderings
	"RESOLUTION": 1024,  # resolution of resulting renders
	
	"LIGHT_SETUP": 'fixed', # none/fixed/colocated

	"COLOR_DEPTH": 8,  # 16
	"FORMAT": 'PNG',  # 'PNG'/OPEN_EXR'/'HDR' # use 16/32 bit color depth with OPEN_EXR format
	"CYCLES_SAMPLES": 10000,
	"CYCLES_MAX_BOUNCES": 20,

	# "CAM_DISTANCE": 1.0,
	"CAM_HEMISPHERE_ANGLES": [-10, 80],  # in degrees
	"RANDOM_VIEWS": True,

	"DEBUG": False,

	"DEPTH_SCALE": 1.4,
	# "UPPER_VIEWS": True,
	"RESULTS_PATH": 'rgb',
}
opts = SimpleNamespace(**options)

