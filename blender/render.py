#!/usr/bin/env python

# imports rendering parameters and runs the creating dataset script

import subprocess, sys
from render_params import opts

cmd = ['blender', opts.PATH_MODEL, '--background', '--python', 'create_dataset.py', '--', opts.DATASET_NAME]
# cmd = ['python', 'test.py']
subprocess.run(cmd, stdout=sys.stdout)


# blender models/toybox.blend --background --python create_dataset.py -- toybox_static_png