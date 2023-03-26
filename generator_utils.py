# Kitti data generator
# Utils for sensors

#https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/ 3D coords to camera coords
#https://github.com/MukhlasAdib/CARLA-2DBBox/blob/1f126343423eb687f1bf88fbf9961bcb9cdb7c65/carla_vehicle_annotator.py#L159

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
from carla import VehicleLightState as vls
from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_2d = np.array([
            point_img[0] / point_img[2],
            point_img[1] / point_img[2],
            point_img[2]
        ])

        return point_2d

def vertices_from_extent(ext):
    """ Extraxts the 8 bounding box vertices relative to (0,0,0)
    https://github.com/carla-simulator/carla/commits/master/Docs/img/vehicle_bounding_box.png 
    8 bounding box vertices relative to (0,0,0)
    """
    return np.array([
        [ext.x,   ext.y,   ext.z],  # Top left front
        [- ext.x,   ext.y,   ext.z],  # Top left back
        [ext.x, - ext.y,   ext.z],  # Top right front
        [- ext.x, - ext.y,   ext.z],  # Top right back
        [ext.x,   ext.y, - ext.z],  # Bottom left front
        [- ext.x,   ext.y, - ext.z],  # Bottom left back
        [ext.x, - ext.y, - ext.z],  # Bottom right front
        [- ext.x, - ext.y, - ext.z]  # Bottom right back
    ])

def create_bb_points(vehicle):
    cords = np.zeros((8, 4))
    extent = vehicle.bounding_box.extent
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    return cords

### Get transformation matrix from carla.Transform object
def get_matrix(transform):
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix 

### Transform coordinate from vehicle reference to world reference
def vehicle_to_world(cords, vehicle):
    bb_transform = carla.Transform(vehicle.bounding_box.location)
    bb_vehicle_matrix = get_matrix(bb_transform)
    vehicle_world_matrix = get_matrix(vehicle.get_transform())
    bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords

### Transform coordinate from world reference to sensor reference
def world_to_sensor(cords, sensor):
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)
    return sensor_cords

### Transform coordinate from vehicle reference to sensor reference
def vehicle_to_sensor(cords, vehicle, sensor):
    world_cord = vehicle_to_world(cords, vehicle)
    sensor_cord = world_to_sensor(world_cord, sensor)
    return sensor_cord

### Draw 2D bounding box (4 vertices) from 3D bounding box (8 vertices) in image
### 2D bounding box is represented by two corner points
def p3d_to_p2d_bb(p3d_bb):
    min_x = np.amin(p3d_bb[:,0])
    min_y = np.amin(p3d_bb[:,1])
    max_x = np.amax(p3d_bb[:,0])
    max_y = np.amax(p3d_bb[:,1])
    p2d_bb = np.array([[min_x,min_y] , [max_x,max_y]])
    return p2d_bb

def get_2d_bbox(actor, camera, K):
    bbox_coord = create_bb_points(actor)
    world_coord = vehicle_to_world(bbox_coord, actor)
    sensor_coord = world_to_sensor(world_coord, camera)
    cords_y_minus_z_x = np.concatenate([sensor_coord[1, :], -sensor_coord[2, :], sensor_coord[0, :]])
    bbox = np.transpose(np.dot(K, cords_y_minus_z_x))
    camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
    bbox_2d = p3d_to_p2d_bb(camera_bbox)
    #print("{} {}".format(bbox_2d[0], bbox_2d[1]))
    return bbox_2d