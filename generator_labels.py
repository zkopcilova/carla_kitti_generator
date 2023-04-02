# Kitti data generator
# Label class and function

# https://github.com/enginBozkurt/carla-training-data - Older Carla version
# https://github.com/jedeschaud/kitti_carla_simulator - Autopilot, generates different lidar format and no labels
# https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d - KITTI 3D

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from generator_utils import *
from math import pi

TYPES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc', 'DontCare']

"""
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
"""

"""
Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %
"""


class Label_Row():
    def __init__(self, type=None, bbox=None, dimensions=None, location=None, rotation_y=None, extent=None):
        self.type = type
        self.truncated = 0
        self.occluded = 0
        self.alpha = 0
        self.bbox = []
        self.bbox_str = ''
        self.dimensions = dimensions
        self.location = location
        self.rotation_y = rotation_y
        
    def set_type(self, type: str):
        assert type in TYPES, "Type is not in KITTI format."
        self.type = type

    def set_truncated(self, truncated: float):
        assert 0 <= truncated <= 1, "Wrong format. Truncated must be a float between 0 and 1."
        self.truncated = truncated

    def set_occluded(self, occlusion: float):
        occluded = 0
        if 0.4 < occlusion <= 0.5:
            occluded = 1
        elif occlusion > 0.5:
            occluded = 2
        self.occluded = occluded

    def set_alpha(self, alpha: float):
        #assert -pi <= alpha <= pi
        self.alpha = alpha

    def set_bbox(self, bbox):
        assert len(bbox) == 4
        self.bbox_str = "{:.2f} {:.2f} {:.2f} {:.2f}".format(bbox[0], bbox[1], bbox[2], bbox[3])
        self.bbox = bbox
        if (bbox[2] - bbox[0]) < 25:
            self.set_type('DontCare')
        self.calculate_truncation(bbox)

    def set_dimensions(self, carla_extent):
        self.dimensions = "{:.2f} {:.2f} {:.2f}".format(2*carla_extent.z, 2*carla_extent.x, 2*carla_extent.y)

    def set_location(self, carla_location, carla_extent_z):
        # Carla:  X   Y   Z
        # KITTI: -X  -Y   Z
        x = - carla_location.x
        y = - carla_location.y
        z = carla_location.z

        if self.type == 'Pedestrian':
            y += carla_extent_z

        self.location = "{:.2f} {:.2f} {:.2f}".format(x, y, z)
    
    def set_rotation_y(self, rotation_y: float):
        assert -pi <= rotation_y <= pi
        self.rotation_y = rotation_y

    def row_to_str(self):
        str = "{} {:.2f} {} {:.2f} {} {} {} {:.2f}".format(self.type, self.truncated, self.occluded, self.alpha, self.bbox_str, self.dimensions, self.location, self.rotation_y)
        return str
    
    def calculate_truncation(self, bbox):        
        bbox_area = calculate_area(bbox[2], bbox[0], bbox[3], bbox[1])
        visible_area = calculate_area(crop_to_range(bbox[2], 0, IMAGE_W),
                                      crop_to_range(bbox[0], 0, IMAGE_W),
                                      crop_to_range(bbox[3], 0, IMAGE_H),
                                      crop_to_range(bbox[1], 0, IMAGE_H))
        self.set_truncated(1 - visible_area/bbox_area)
        if self.truncated > 0.5:
            self.set_type('DontCare')


