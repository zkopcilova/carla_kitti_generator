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
import logging
import math
import random
from queue import Queue
from queue import Empty

from math import *
from generator_labels import *
from generator_bbox import *

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
            label_str += ("\n"+ row.row_to_str())
        f.write(label_str)

def save_calib_sample(current_frame, camera, lidar):
    filename = CALIB_DATA.format(current_frame)
    P0 = build_projection_matrix(IMAGE_W, IMAGE_H, FOV)
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order='C')
    R0 = np.identity(3)
    # Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne
    TR_velodyne = np.array([[0, 1, 0],
                   [0, 0, -1],
                   [1, 0, 0]])
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
    lidar_array = [[point[0], point[1], point[2], 1.0]
                    for point in point_cloud]
    lidar_array = np.array(lidar_array).astype(np.float32)
    lidar_array.tofile(LIDAR_DATA.format(current_frame))

def save_image_sample(current_frame, im_array):
    image = Image.fromarray(im_array)
    image.save(IMAGE_DATA.format(current_frame))

def sensor_callback(data, queue):
    queue.put(data)

def get_detected_objects(world, camera, depth_image):
    rows = []
    visible_actors =  filter_angle_distance(world.get_actors(), camera)
    for actor in visible_actors:
        bp = world.get_blueprint_library().filter(actor.type_id)
        if (bp):
            dist = actor.get_transform().location.distance(actor.get_transform().location)
            if (bp[0].has_tag('vehicle') or bp[0].has_tag('pedestrian')) and dist < 50:
                row = create_label_row(actor, bp[0], camera, depth_image)
                rows.append(row) if row else None            
    return rows

def create_label_row(actor, actor_bp, camera, depth_image):
    agent_type, extent, location = agent_attributes(actor, actor_bp)
    #occlusion = calculate_occlusion()
    bbox, occlusion = get_2d_bbox(actor, camera, depth_image)
    rotation = deg_to_rad(actor.get_transform().rotation.yaw)
    obs_angle = get_observation_angle(actor, camera)
    alpha = rotation - obs_angle


    row = Label_Row()
    row.set_alpha(alpha)
    row.set_type(agent_type)
    row.set_bbox(bbox)

    if occlusion == 1 or row.truncated == 1 or (bbox[3]-bbox[1] < 20):
        return None 

    row.set_dimensions(extent)
    row.set_location(location, extent.z)
    row.set_occluded(occlusion)
    row.set_rotation_y(rotation)   

    return row
    
def agent_attributes(actor, actor_bp):
    if actor_bp.id in VANS:
        type = 'Van'
    elif actor_bp.id in TRUCKS:
        type = 'Truck'
    elif actor_bp.id == BIKE:
        type = 'Cyclist'
    else:
        if actor_bp.has_tag('vehicle'):
                type = 'Car'
        elif actor_bp.has_tag('pedestrian'):
            type = 'Pedestrian'
    extent = actor.bounding_box.extent
    location = actor.get_location()

    return type, extent, location

def generator_loop(args):
    starting_frame = args.starting_frame
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(False)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    vehicle = None
    camera = None
    lidar = None
    depth_camera = None

    try:
        # Configure blueprints
        blueprints = bp_lib.filter("vehicle.audi*")
        vehicle_bp = random.choice(blueprints)
        vehicle_bp.set_attribute('role_name', 'autopilot')
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast")[0]
        depth_camera_bp = bp_lib.filter("sensor.camera.depth")[0]
    
        camera_bp.set_attribute("image_size_x", str(IMAGE_W))
        camera_bp.set_attribute("image_size_y", str(IMAGE_H))

        depth_camera_bp.set_attribute("image_size_x", str(IMAGE_W))
        depth_camera_bp.set_attribute("image_size_y", str(IMAGE_H))

        lidar_bp.set_attribute('upper_fov', "7.0")
        lidar_bp.set_attribute('lower_fov', "-16.0")
        lidar_bp.set_attribute('channels', "64.0")
        lidar_bp.set_attribute('range', "100.0")
        lidar_bp.set_attribute('points_per_second', "720000")

        # Spawn blueprints
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            transform=world.get_map().get_spawn_points()[5])
        vehicle.set_autopilot(True)
        camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=1.3, z=1.6)),
            attach_to=vehicle)
        lidar = world.spawn_actor(
            blueprint=lidar_bp,
            transform=carla.Transform(carla.Location(x=1.0, z=1.6)),
            attach_to=vehicle)
        depth_camera = world.spawn_actor(
            blueprint=depth_camera_bp,
            transform=carla.Transform(carla.Location(x=1.3, z=1.6)),
            attach_to=vehicle)

        time.sleep(2)

        # Sensor data
        image_queue = Queue()
        lidar_queue = Queue()
        depth_queue = Queue()

        camera.listen(lambda data: sensor_callback(data, image_queue))
        lidar.listen(lambda data: sensor_callback(data, lidar_queue))
        depth_camera.listen(lambda data: sensor_callback(data, depth_queue))

        for frame in range(args.frames):
            frame_num = starting_frame + frame
            cnt = 0
            detected = []

            while cnt < 60:
                world.tick()
                cnt += 1
                try:
                    # Get data when it's received.
                    image_data = image_queue.get(False)
                    lidar_data = lidar_queue.get(False)
                    depth_data = depth_queue.get(False)
                except Empty:
                    print("[Warning] Some sensor data has been missed")
                    continue

            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            transform_matrix, rot = lidar_to_world_rot(vehicle, lidar)

            p_cloud_size = len(lidar_data)
            p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

            #p_cloud = transform_matrix * p_cloud.T
            #p_cloud = p_cloud.T * rot

            local_lidar_points = np.array(p_cloud[:, :3]).T
            print(local_lidar_points.shape)

            # Add an extra 1.0 at the end of each 3d point so it becomes of
            # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
            local_lidar_points = np.r_[
                local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
            print(local_lidar_points.shape)

            # This (4, 4) matrix transforms the points from lidar space to world space.
            lidar_2_world = lidar.get_transform().get_matrix()
            #print(lidar_2_world.shape)

            # Transform the points from lidar space to world space.
            world_points = np.dot(lidar_2_world, local_lidar_points)
            print(world_points.shape)

            # This (4, 4) matrix transforms the points from world to sensor coordinates.
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
            

            # Transform the points from world space to camera space.
            sensor_points = np.dot(world_2_camera, world_points)
            print(sensor_points.shape)
            detected = get_detected_objects(world, camera, depth_data)
            if detected == []:
                frame -= 1
                continue

            save_image_sample(frame_num, im_array)
            save_lidar_sample(frame_num, sensor_points.T)
            save_label_sample(frame_num, detected)
            save_calib_sample(frame_num, camera, lidar)

            
    finally:
        # Apply the original settings when exiting.
        world.apply_settings(original_settings)

        # Destroy the actors in the scene.
        if camera:
            camera.destroy()
        if lidar:
            lidar.destroy()
        if depth_camera:
            depth_camera.destroy()
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
        '-f', '--frames',
        metavar='N',
        default=10,
        type=int,
        help='number of frames to record (default: 10)')
    argparser.add_argument(
        '-s', '--starting_frame',
        metavar='S',
        default=0,
        type=int,
        help='frame number to begin with (default: 0)')
    args = argparser.parse_args()

    try:
        generator_loop(args)
    
    except KeyboardInterrupt:
        logging.info('\nCancelled by user. Bye!')

if __name__ == '__main__':

    main()       