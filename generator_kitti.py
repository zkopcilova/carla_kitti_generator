# Kitti data generator
# Goal: use manual control in simulator to generate synthetic data in Kitti format

# https://github.com/enginBozkurt/carla-training-data - Older Carla version
# https://github.com/jedeschaud/kitti_carla_simulator - Autopilot, generates different lidar format and no labels
# https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d - KITTI 3D
# https://github.com/carla-simulator/data-collector/blob/master/carla/client.py - READ DATA ALTERNATIVE

from __future__ import print_function

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
from carla import ColorConverter as cc
from carla import Client 

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
from queue import Queue
from queue import Empty
from matplotlib import cm

from math import pi
from generator_labels import Label_Row
from generator_utils import *

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')


""" OUTPUT FOLDERS """
PHASE = "training"
OUTPUT_FOLDER = os.path.join(PHASE)
folders = ['calib', 'image_2', 'label_2', 'velodyne']


def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)

for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)

""" PATHS """
CALIB_DATA = os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt')
LIDAR_DATA = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
LABEL_DATA = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt')
IMAGE_DATA = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png')

def save_label_sample(current_frame, label_rows):
    filename = LABEL_DATA.format(current_frame)
    label_str=""
    with open(filename, 'w') as f:
        for row in label_rows:
            label_str += ("\n"+row.row_to_str())
        f.write(label_str)

def save_calib_sample(current_frame, K):
    filename = CALIB_DATA.format(current_frame)
    P0 = K
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order='C')
    R0 = np.identity(3)
    TR_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
    # Add translation vector from velo to camera. This is 0 because the position of camera and lidar is equal in our configuration.
    TR_velodyne = np.column_stack((TR_velodyne, np.array([0, 0, 0])))
    TR_imu_to_velo = np.identity(3)
    TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))

    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten('C').squeeze()))))

    # All matrices are written on a line with spacing
    with open(filename, 'w') as f:
        for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", TR_velodyne)
        write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)
    logging.info("Wrote all calibration matrices to %s", filename)

def save_lidar_sample(current_frame, point_cloud):
    lidar_array = [[point[0], -point[1], point[2], 1.0]
                    for point in point_cloud]
    lidar_array = np.array(lidar_array).astype(np.float32)
    lidar_array.tofile(LIDAR_DATA.format(current_frame))

def save_image_sample(current_frame, im_array):
    image = Image.fromarray(im_array)
    image.save(IMAGE_DATA.format(current_frame))

def sensor_callback(data, queue):
    queue.put(data)

def get_detected_objects(world, camera, K):
    rows = []
    all_actors = world.get_actors()
    for actor in all_actors:
        print(actor.type_id)
        bp = world.get_blueprint_library().filter(actor.type_id)
        if (bp):
            dist = actor.get_transform().location.distance(actor.get_transform().location)
            if (bp[0].has_tag('vehicle') or bp[0].has_tag('pedestrian')) and dist < 50:
                rows.append(create_label_row(actor, bp[0], camera, K))
                print(actor.type_id)                    
    return rows

def create_label_row(actor, actor_bp, camera, K):
    agent_type, extent, location = agent_attributes(actor, actor_bp)
    #truncated = 0.0
    #occlusion = calculate_occlusion()
    bbox = get_2d_bbox(actor, camera, K)  
    rotation_actor = actor.get_transform().rotation.yaw
    rotation_camera = camera.get_transform().rotation.yaw
    rotation = (rotation_camera - rotation_actor) * math.pi / 180
           

    row = Label_Row()
    row.set_type(agent_type)
    row.set_bbox(bbox)
    row.set_dimensions(extent)
    row.set_location(location, extent.z)
    if (- pi <= rotation <= pi):
        row.set_rotation_y(rotation)    

    print("row")
    return row
    
def agent_attributes(actor, actor_bp):
    if actor_bp.has_tag('vehicle'):
        if (actor_bp.get_attribute('number_of_wheels').as_int() == 2):
            type = 'Cyclist'
        else:
            type = 'Car'
    elif actor_bp.has_tag('pedestrian'):
        type = 'Pedestrian'
    extent = actor.bounding_box.extent
    location = actor.get_location()

    return type, extent, location

def calculate_occlusion():
    return 0

def get_bbox_coordinates():
    return [0,0,0,0]

def generator_loop(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    #traffic_manager = client.get_trafficmanager(8000)
    #traffic_manager.set_synchronous_mode(False)

    original_settings = world.get_settings()
    #settings = world.get_settings()
    #settings.synchronous_mode = True
    #settings.fixed_delta_seconds = 0.05
    #world.apply_settings(settings)

    vehicle = None
    camera = None
    lidar = None

    try:
        # Configure blueprints
        blueprints = bp_lib.filter("vehicle.audi*")
        vehicle_bp = random.choice(blueprints)
        vehicle_bp.set_attribute('role_name', 'autopilot')
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast")[0]
    
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))

        lidar_bp.set_attribute('upper_fov', "7.0")
        lidar_bp.set_attribute('lower_fov', "-16.0")
        lidar_bp.set_attribute('channels', "64.0")
        lidar_bp.set_attribute('range', "120.0")
        lidar_bp.set_attribute('points_per_second', "100000")

        # Spawn blueprints
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            transform=world.get_map().get_spawn_points()[15])
        vehicle.set_autopilot(True)
       # traffic_manager.vehicle_percentage_speed_difference(vehicle, -20.0)
        camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=1.6, z=1.6)),
            attach_to=vehicle)
        lidar = world.spawn_actor(
            blueprint=lidar_bp,
            transform=carla.Transform(carla.Location(x=1.0, z=1.8)),
            attach_to=vehicle)
        
        #spawn_points = world.get_map().get_spawn_points()
        #destination = random.choice(spawn_points).location
        #vehicle.set_destination(destination)

        time.sleep(2)

        # Sensor data
        image_queue = Queue()
        lidar_queue = Queue()

        camera.listen(lambda data: sensor_callback(data, image_queue))
        lidar.listen(lambda data: sensor_callback(data, lidar_queue))

        for frame in range(args.frames):
            cnt = 0
            while cnt < 10:
                world.tick()
                world.wait_for_tick()
                cnt += 1
                try:
                    # Get data when it's received.
                    image_data = image_queue.get(True, 5.0)
                    lidar_data = lidar_queue.get(True, 5.0)
                except Empty:
                    print("[Warning] Some sensor data has been missed")
                    continue

            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            p_cloud_size = len(lidar_data)
            p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

            # Get the world to camera matrix
            w2c = np.array(camera.get_transform().get_inverse_matrix())

            # Get the attributes from the camera
            image_w = camera_bp.get_attribute("image_size_x").as_int()
            image_h = camera_bp.get_attribute("image_size_y").as_int()
            fov = camera_bp.get_attribute("fov").as_float()

            # Calculate the camera projection matrix to project from 3D -> 2D
            K = build_projection_matrix(image_w, image_h, fov)

            detected = get_detected_objects(world, camera, K)

            save_image_sample(frame, im_array)
            #save_lidar_sample(frame, p_cloud)
            save_label_sample(frame, detected)
            save_calib_sample(frame, K)

            
    finally:
        # Apply the original settings when exiting.
        world.apply_settings(original_settings)

        # Destroy the actors in the scene.
        if camera:
            camera.destroy()
        if lidar:
            lidar.destroy()
        if vehicle:
            vehicle.destroy()

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=10,
        type=int,
        help='number of frames to record (default: 10)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        generator_loop(args)
    
    except KeyboardInterrupt:
        logging.info('\nCancelled by user. Bye!')

if __name__ == '__main__':

    main()       