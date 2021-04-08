# yeeeey

from types import SimpleNamespace


options = {
	"DEBUG": False,

	"PATH_MODEL": 'models/rocket.blend',
	"DATASET_NAME": 'rocket_png',

	"VOXEL_NUMS": 64,  # 512
	"VIEWS": 50, # 200
	"RESOLUTION": 800,
	"COLOR_DEPTH": 8,  # 16
	"FORMAT": 'PNG',  # 'PNG'/OPEN_EXR'/'HDR' # use 16/32 bit color depth with OPEN_EXR format

	"CYCLES_SAMPLES": 500,
	"CIRCLE_FIXED_START": (.3,0,0),
	"LIGHT_SETUP": 'fixed', # none/fixed/colocated
	# "CAM_INIT_LOCATION": (5, -3, 5),  # or None (4, -4, 4)
	# "CAM_DISTANCE": 5,

	"DEPTH_SCALE": 1.4,
	"RANDOM_VIEWS": True,
	"UPPER_VIEWS": True,
	"RESULTS_PATH": 'rgb',
}
opts = SimpleNamespace(**options)

