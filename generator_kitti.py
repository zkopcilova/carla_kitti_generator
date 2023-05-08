"""
    Bachelor thesis
    Topic:        Using synthetic data for improving detection of cyclists and pedestrians in autonomous driving
    Author:       Zuzana Kopčilová
    Institution:  Brno University of Technology, Faculty of Information Technology
    Date:         05/2023
"""

"""
    Main dataset creation script
"""

""" Script is based on example CARLA scripts (mainly lidar_to_camera.py) and existing works focusing on
    synthetic dataset creation.
    
    [1] https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/lidar_to_camera.py 
    [2] https://github.com/enginBozkurt/carla-training-data
    [3] https://github.com/jedeschaud/kitti_carla_simulator
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

from matplotlib import cm
VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


""" OUTPUT FOLDERS """
PHASE = "training"
OUTPUT_FOLDER = os.path.join(PHASE)
folders = ['calib', 'image_2', 'label_2', 'velodyne']

""" PATHS """
CALIB_DATA = os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt')
LIDAR_DATA = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
LABEL_DATA = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt')
IMAGE_DATA = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png')

""" Folder preparation """
def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)

for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)


""" Methods for saving data samples """

def save_label_sample(current_frame, label_rows):
    filename = LABEL_DATA.format(current_frame)
    label_str=""
    with open(filename, 'w') as f:
        for row in label_rows:
            label_str += row.row_to_str()
            label_str +="\n" if row != label_rows[-1] else ""
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

    with open(filename, 'w') as f:
        for i in range(4):  # Only one camera is used
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

def get_detected_objects(world, camera, depth_image):
    rows = []
    visible_actors =  filter_angle_distance(world.get_actors(), camera)
    for actor in visible_actors:
        bp = world.get_blueprint_library().filter(actor.type_id)
        if (bp):
            if (bp[0].has_tag('vehicle') or bp[0].has_tag('pedestrian')):
                row = create_label_row(actor, bp[0], camera, depth_image)
                rows.append(row) if row else None            
    return rows

def create_label_row(actor, actor_bp, camera, depth_image):
    agent_type, extent, location = agent_attributes(actor, actor_bp)
    bbox, occlusion = get_2d_bbox(actor, camera, depth_image)
    rotation = - deg_to_rad(actor.get_transform().rotation.yaw)
    obs_angle = get_observation_angle(actor, camera)
    alpha = rotation - obs_angle

    # convert location to camera coordinates
    loc = np.array([location.x, location.y, location.z, 1])
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    loc = np.dot(world_2_camera, loc.T)

    row = Label_Row()
    row.set_alpha(alpha)
    row.set_type(agent_type)
    row.set_bbox(bbox)

    if occlusion == 1 or row.truncated == 1 or (bbox[3]-bbox[1] < 20):
        return None 

    row.set_dimensions(extent)

    #    UE4    ->   KITTI
    # (x, y, z) -> (y, -z, x)
    row.set_location(loc[1], -loc[2], loc[0], extent.z)
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
        lidar_bp.set_attribute('lower_fov', "-20.0")
        lidar_bp.set_attribute('channels', "64.0")
        lidar_bp.set_attribute('range', "100.0")
        lidar_bp.set_attribute('points_per_second', "3000000")
        lidar_bp.set_attribute('rotation_frequency','20')

        # Spawn blueprints
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            transform=world.get_map().get_spawn_points()[5])
        vehicle.set_autopilot(True)
        camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=1.2, z=1.6)),
            attach_to=vehicle)
        lidar = world.spawn_actor(
            blueprint=lidar_bp,
            transform=carla.Transform(carla.Location(x=1.3, z=1.8)),
            attach_to=vehicle)
        depth_camera = world.spawn_actor(
            blueprint=depth_camera_bp,
            transform=carla.Transform(carla.Location(x=1.2, z=1.6)),
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
                world.tick(40)
                cnt += 1
                try:
                    # Get data when it's received.
                    image_data = image_queue.get(False)
                    lidar_data = lidar_queue.get(False)
                    depth_data = depth_queue.get(False)
                except Empty:
                    print("[Warning] Some sensor data has been missed")
                    continue

            """ Beginning of section taken directly from source [1] """
            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            p_cloud_size = len(lidar_data)
            p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))


            local_lidar_points = np.array(p_cloud[:, :3]).T

            local_lidar_points = np.r_[
                local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

            # This (4, 4) matrix transforms the points from lidar space to world space.
            lidar_2_world = lidar.get_transform().get_matrix()
            # Transform the points from lidar space to world space.
            world_points = np.dot(lidar_2_world, local_lidar_points)
            # This (4, 4) matrix transforms the points from world to sensor coordinates.
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
            # Transform the points from world space to camera space.
            sensor_points = np.dot(world_2_camera, world_points)

            detected = get_detected_objects(world, camera, depth_data)
            if detected == []:
                frame -= 1
                continue

            projected = np.copy(im_array)
            
            point_in_camera_coords = np.array([
                sensor_points[1],
                sensor_points[2] * -1,
                sensor_points[0]])
            
            K = build_projection_matrix(IMAGE_W, IMAGE_H, FOV)

            # Finally we can use our K matrix to do the actual 3D -> 2D.
            points_2d = np.dot(K, point_in_camera_coords)

            # Remember to normalize the x, y values by the 3rd value.
            points_2d = np.array([
                points_2d[0, :] / points_2d[2, :],
                points_2d[1, :] / points_2d[2, :],
                points_2d[2, :]])

            # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
            # contains all the y values of our points. In order to properly
            # visualize everything on a screen, the points that are out of the screen
            # must be discarted, the same with points behind the camera projection plane.
            points_2d = points_2d.T
            intensity = intensity.T
            points_in_canvas_mask = \
                (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < IMAGE_W) & \
                (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < IMAGE_H) & \
                (points_2d[:, 2] > 0.0)
            points_2d = points_2d[points_in_canvas_mask]
            intensity = intensity[points_in_canvas_mask]

            # Extract the screen coords (uv) as integers.
            u_coord = points_2d[:, 0].astype(np.int)
            v_coord = points_2d[:, 1].astype(np.int)

            # Since at the time of the creation of this script, the intensity function
            # is returning high values, these are adjusted to be nicely visualized.
            intensity = 4 * intensity - 3
            color_map = np.array([
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T

            dot_extent = 2
            # Draw the 2d points on the image as squares of extent args.dot_extent.
            for i in range(len(points_2d)):
                # I'm not a NumPy expert and I don't know how to set bigger dots
                # without using this loop, so if anyone has a better solution,
                # make sure to update this script. Meanwhile, it's fast enough :)
                projected[
                    v_coord[i]-dot_extent : v_coord[i]+dot_extent,
                    u_coord[i]-dot_extent : u_coord[i]+dot_extent] = color_map[i]

            # Save the image using Pillow module.
            image = Image.fromarray(projected)
            image.save("_out/%06d.png" % frame_num)

            """ End of section taken directly from source [1] """

            save_image_sample(frame_num, im_array)
            save_lidar_sample(frame_num, sensor_points.T)
            save_label_sample(frame_num, detected)
            save_calib_sample(frame_num, camera, lidar)
            print(frame_num)

            
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