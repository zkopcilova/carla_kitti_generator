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
from itertools import product
from math import pi

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

IMAGE_W =  1280
IMAGE_H = 720
FOV = 90

VANS = ['vehicle.mercedes.sprinter', 'ford.ambulance']
TRUCKS = ['vehicle.carlamotors.carlacola', 'vehicle.carlamotors.firetruck']
BIKE = 'vehicle.bh.crossbike'

def crop_to_range(value, min, max):
    if value <= min:
        value = min
    elif value >= max:
        value = max - 1
    return value

def calculate_area(x_min, x_max, y_min, y_max):
    return abs((x_max - x_min)*(y_max - y_min))

def deg_to_rad(degrees):
    return degrees * pi / 180

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    
def point_is_occluded(vertex_x, vertex_y, vertex_depth, depth_map):
    """ Checks whether or not the four pixels directly around the given point has less depth than the given vertex depth
        If True, this means that the point is occluded.
    """
    x = int(vertex_x)
    y = int(vertex_y)
    neigbours = product((1, -1), repeat=2)
    is_occluded = []
    for dx, dy in neigbours:
        depth_x = crop_to_range(x+dx, 0, IMAGE_W)
        depth_y = crop_to_range(y+dy, 0, IMAGE_H)
        # If the depth map says the pixel is closer to the camera than the actual vertex
        if depth_map[depth_y, depth_x] < vertex_depth:
            is_occluded.append(True)
        else:
            is_occluded.append(False)
    # Only say point is occluded if all four neighbours are closer to camera than vertex
    return all(is_occluded)