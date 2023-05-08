"""
    Bachelor thesis
    Topic:        Using synthetic data for improving detection of cyclists and pedestrians in autonomous driving
    Author:       Zuzana Kopčilová
    Institution:  Brno University of Technology, Faculty of Information Technology
    Date:         05/2023
"""

"""
    Simulator environment setup, spawning actors
    ----------------------
    
    Script is based on tutorial script available with the CARLA simulator:
    https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/generate_traffic.py
"""

import glob
import os
import sys
import time
import argparse
import logging
from numpy import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import VehicleLightState as vls


# Commands
SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
        return bps
    except:
        print("Generation is not valid, no actor will be spawned.")
        return []

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
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '-w','--walkers',
        default=300,
        type=int,
        help='Number of walkers to spawn (default: 300)')
    argparser.add_argument(
        '-v','--vehicles',
        default=15,
        type=int,
        help='Number of vehicles to spawn (default: 15)')
    argparser.add_argument(
        '-c','--cyclists',
        default=25,
        type=int,
        help='Number of cyclists to spawn (default: 25)')
    argparser.add_argument(
        '-t','--town',
        default=2,
        type=int,
        help='Which map to generate - values 1-5 (default: 2)')

    args = argparser.parse_args()

    vehicles_list = []
    walkers_list = []
    walkers_controllers = []
    
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(int(time.time()))

    n_vehicles = n_max = args.vehicles + args.cyclists
    n_cyclists = args.cyclists
    n_walkers = args.walkers

    try:
        # --------------
        # World
        # --------------

        map = '/Game/Carla/Maps/Town0' + str(args.town)
        world = client.load_world(map)
        settings = world.get_settings()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        traffic_manager.global_percentage_speed_difference(80.0)
        traffic_manager.set_synchronous_mode(True)

        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
        else:
            synchronous_master = False

        world.apply_settings(settings)

        # --------------
        # Get actor blueprints
        # Filter out motorcycles - not a Kitti category
        # --------------
        blueprints = get_actor_blueprints(world, "vehicle.*", "All")
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        blueprintsWalkers = get_actor_blueprints(world, "walker.pedestrian.*", "2")

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        n_spawn_points = len(spawn_points)

        if n_vehicles > n_spawn_points:
            n_vehicles = n_spawn_points

        # --------------
        # Spawn vehicles
        # -------------- 
        batch = []
        i = 0

        for n_vehicles, transform in enumerate(spawn_points):
            if i >= n_max:
                break
            if i < n_cyclists:
                blueprint = world.get_blueprint_library().find('vehicle.bh.crossbike')
            else:
                blueprint = random.choice(blueprints)
            
            i+=1
            
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            blueprint.set_attribute('role_name', 'autopilot')

            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # turn on lights
        for vehicle in world.get_actors(vehicles_list):
            vehicle.set_light_state(vls.LowBeam)


        # -------------
        # Spawn Walkers
        # -------------
        percentageRunning = 0.1 
        percentageCrossing = 0.5

        # locations to spawn
        spawn_points = []
        for i in range(n_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentageRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        # walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id

        # put together the walkers and controllers
        for i in range(len(walkers_list)):
            walkers_controllers.append(walkers_list[i]["con"])
            walkers_controllers.append(walkers_list[i]["id"])
        all_actors = world.get_actors(walkers_controllers)

        # wait for tick to synchronize changes
        if not synchronous_master:
            world.wait_for_tick() 
        else:
            world.tick()
            
        # set target location and crossing the road
        world.set_pedestrians_cross_factor(percentageCrossing)
        for i in range(0, len(walkers_controllers), 2):
            all_actors[i].start()
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        # print info to user
        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))
        
        # move actors out of spawn points
        cnt = 0
        while cnt < 100:
            world.tick()
            cnt += 1

        print("waiting for tick now")
        while True:
            world.wait_for_tick(60.0)

    finally:

        if synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        # destroy cars and cyclists
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # destroy walkers + controllers
        for i in range(0, len(walkers_controllers), 2):
            all_actors[i].stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in walkers_controllers])

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')