# Kitti data generator
# Utils for sensors

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

IMAGE_W =  1280
IMAGE_H = 720

def crop_to_range(value, min, max):
    if value < min:
        value = min
    elif value > max:
        value = max
    return value

def calculate_area(x_min, x_max, y_min, y_max):
    return abs((x_max - x_min)*(y_max - y_min))