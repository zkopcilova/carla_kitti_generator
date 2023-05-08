"""
    Bachelor thesis
    Topic:        Using synthetic data for improving detection of cyclists and pedestrians in autonomous driving
    Author:       Zuzana Kopčilová
    Institution:  Brno University of Technology, Faculty of Information Technology
    Date:         05/2023
"""

"""
    Utility methods and constants for other scripts
"""

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

def angle_between(v1, v2):
    """ Returns angle between vectors 'v1' and 'v2' in radians
    """
    v1_u = np.linalg.norm(v1)
    v2_u = np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    
def point_is_occluded(vertex_x, vertex_y, vertex_depth, depth_map):
    """ Checks if the four pixels directly around the given point are closer to camera
        than the evaluated vertex
    """
    x = int(vertex_x)
    y = int(vertex_y)
    neigbours = product((1, -1), repeat=2)
    is_occluded = []
    for dx, dy in neigbours:
        depth_x = crop_to_range(x+dx, 0, IMAGE_W)
        depth_y = crop_to_range(y+dy, 0, IMAGE_H)

        if depth_map[depth_y, depth_x] < vertex_depth:
            is_occluded.append(True)
        else:
            is_occluded.append(False)

    # True if all four neighbours are closer to camera than vertex
    return all(is_occluded)

def angle_limit(a):
    """ Limits angle to values in range [-pi/2; pi/2] 
    """
    while a > pi/2:
        a -= pi
    while a < - pi/2:
        a += pi
    return a