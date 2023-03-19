# Kitti data generator
# Goal: use manual control in simulator to generate synthetic data in Kitti format

# https://github.com/enginBozkurt/carla-training-data - Older Carla version
# https://github.com/jedeschaud/kitti_carla_simulator - Autopilot, generates different lidar format and no labels
# https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d - KITTI 3D

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

from generator_classes import Label_Row

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
    
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
    with open(filename, 'w') as f:
        label_str = "\n".join([str(row) for row in label_rows if row])
        f.write(label_str)

#def save_calib_sample():

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

def get_detected_objects(world, measurements):
    for agent in measurements.non_player_agents:
        if agent.hasField('vehicle') or agent.hasField('pedestrian'):
            create_label_row(agent)

def create_label_row(world, agent):
    agent_type, agent_transform, bbox_transform, extent, location = agent_attributes(world, agent)
    #truncated = 0.0
    #occlusion = calculate_occlusion()
    bbox = get_bbox_coordinates()
    rotation = 0

    row = Label_Row()
    row.set_type(agent_type)
    row.set_bbox(bbox)
    row.set_dimensions(extent)
    row.set_location(location, extent.z)
    row.set_rotation_y(rotation)
    return row
    


def agent_attributes(world, agent):
    if agent.hasField('vehicle'):
        actor = world.get_actors().find(agent.id)
        if (actor.get_attribute('number_of_wheels') == 2):
            type = 'Cyclist'
        else:
            type = 'Vehicle'
        agent_transform = agent.vehicle.transform
        bbox_transform = agent.vehicle.bounding_box.transform
        ext = agent.vehicle.bounding_box.extent
        location = agent.vehicle.transform.location
    elif agent.hasField('pedestrian'):
        type = 'Pedestrian'
        agent_transform = agent.pedestrian.transform
        bbox_transform = agent.pedestrian.bounding_box.transform
        extent = agent.pedestrian.bounding_box.extent
        location = agent.pedestrian.transform.location
    return type, agent_transform, bbox_transform, extent, location

def calculate_occlusion():
    return 0

def get_bbox_coordinates():
    return [0,0,0,0]

def generator_loop(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 3.0
    world.apply_settings(settings)

    vehicle = None
    camera = None
    lidar = None

    try:
        # Configure blueprints
        blueprints = bp_lib.filter("vehicle.*")
        vehicle_bp = random.choice(blueprints)
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast_semantic")[0]
    
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))

        lidar_bp.set_attribute('upper_fov', "7.0")
        lidar_bp.set_attribute('lower_fov', "-16.0")
        lidar_bp.set_attribute('channels', "64.0")
        lidar_bp.set_attribute('range', "120.0")
        lidar_bp.set_attribute('points_per_second', "1300000")

        # Spawn blueprints
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            transform=world.get_map().get_spawn_points()[15])
        vehicle.set_autopilot(True)
        camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=1.6, z=1.6)),
            attach_to=vehicle)
        lidar = world.spawn_actor(
            blueprint=lidar_bp,
            transform=carla.Transform(carla.Location(x=1.0, z=1.8)),
            attach_to=vehicle)
        
        # Sensor data
        image_queue = Queue()
        lidar_queue = Queue()

        camera.listen(lambda data: sensor_callback(data, image_queue))
        lidar.listen(lambda data: sensor_callback(data, lidar_queue))

        for frame in range(args.frames):
            world.tick()
            time.sleep(10)
            measurements, sensor_data = client.read_data() 

            try:
                # Get data when it's received.
                image_data = image_queue.get(True, 1.0)
                lidar_data = lidar_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            p_cloud_size = len(lidar_data)
            p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

            detected = get_detected_objects(world, measurements)

            save_image_sample(frame, im_array)
            save_lidar_sample(frame, p_cloud)
        
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