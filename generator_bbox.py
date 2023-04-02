# Kitti data generator
# Functions for bboxes and filtering actors in camera fov

#https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/ 3D coords to camera coords
#https://github.com/MukhlasAdib/CARLA-2DBBox/blob/1f126343423eb687f1bf88fbf9961bcb9cdb7c65/carla_vehicle_annotator.py#L159

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
#from carla import VehicleLightState as vls
#from carla import ColorConverter as cc

from generator_utils import *

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

### Use this function to convert depth image (carla.Image) to a depth map in meter
def extract_depth(depth_img):
    depth_img.convert(carla.ColorConverter.Depth)
    depth_meter = np.array(depth_img.raw_data).reshape((depth_img.height,depth_img.width,4))[:,:,0] * 1000 / 255
    return depth_meter

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

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
    p2d_bb = np.array([min_x,min_y,max_x,max_y])
    return p2d_bb

### Get numpy 2D array of vehicles' location and rotation from world reference, also locations from sensor reference
def get_list_transform(vehicles_list, sensor):
    t_list = []
    for vehicle in vehicles_list:
        v = vehicle.get_transform()
        transform = [v.location.x , v.location.y , v.location.z , v.rotation.roll , v.rotation.pitch , v.rotation.yaw]
        t_list.append(transform)
    t_list = np.array(t_list).reshape((len(t_list),6))
    
    transform_h = np.concatenate((t_list[:,:3],np.ones((len(t_list),1))),axis=1)
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    transform_s = np.dot(world_sensor_matrix, transform_h.T).T
    
    return t_list , transform_s

### Remove vehicles that are not in the FOV of the sensor
def filter_angle(vehicles_list, v_transform, v_transform_s):
    v_angle = np.arctan2(v_transform_s[:,1],v_transform_s[:,0]) * 180 / np.pi
    selector = np.array(np.absolute(v_angle) < (int(FOV)/2))
    vehicles_list_f = [v for v, s in zip(vehicles_list, selector) if s]
    v_transform_f = v_transform[selector[:,0],:]
    v_transform_s_f = v_transform_s[selector[:,0],:]
    return vehicles_list_f , v_transform_f , v_transform_s_f

### Remove vehicles that have distance > max_dist from the sensor
def filter_distance(vehicles_list, v_transform, v_transform_s, sensor, max_dist):
    s = sensor.get_transform()
    s_transform = np.array([s.location.x , s.location.y , s.location.z])
    dist2 = np.sum(np.square(v_transform[:,:3] - s_transform), axis=1)
    selector = dist2 < (max_dist**2)
    vehicles_list_f = [v for v, s in zip(vehicles_list, selector) if s]
    v_transform_f = v_transform[selector,:]
    v_transform_s_f = v_transform_s[selector,:] 
    return vehicles_list_f , v_transform_f , v_transform_s_f

### Apply angle and distance filters in one function
def filter_angle_distance(vehicles_list, sensor, max_dist=150):
    vehicles_transform , vehicles_transform_s = get_list_transform(vehicles_list, sensor)
    vehicles_list , vehicles_transform , vehicles_transform_s = filter_distance(vehicles_list, vehicles_transform, vehicles_transform_s, sensor, max_dist)
    vehicles_list , vehicles_transform , vehicles_transform_s = filter_angle(vehicles_list, vehicles_transform, vehicles_transform_s)
    return vehicles_list

def get_2d_bbox(actor, camera, depth_image):
    K = build_projection_matrix(IMAGE_W, IMAGE_H, FOV)
    bbox_coord = create_bb_points(actor)
    world_coord = vehicle_to_world(bbox_coord, actor)
    sensor_coord = world_to_sensor(world_coord, camera)
    cords_y_minus_z_x = np.concatenate([sensor_coord[1, :], -sensor_coord[2, :], sensor_coord[0, :]])
    bbox = np.transpose(np.dot(K, cords_y_minus_z_x))
    camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
    occlusion = calculate_bbox_occlusion(camera_bbox, depth_image)
    bbox_2d = p3d_to_p2d_bb(camera_bbox)
    return bbox_2d, occlusion

def calculate_bbox_occlusion(bbox, depth_image):
    occluded_vertices = 0
    depth_map = extract_depth(depth_image)
    for vertex in bbox:
        if point_is_occluded(vertex[0,0], vertex[0,1], vertex[0,2], depth_map):
            occluded_vertices += 1
    return float(occluded_vertices/8)

def get_observation_angle(vehicle, sensor):
    v = vehicle.get_transform()
    transform = np.array([v.location.x , v.location.y , v.location.z, 1])
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    transform_s = np.dot(world_sensor_matrix, transform.T).T
    v_angle = np.arctan2(transform_s[1,0],transform_s[0,0]) * 180 / np.pi
    v_angle = deg_to_rad(v_angle)
    return v_angle