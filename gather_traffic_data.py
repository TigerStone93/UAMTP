import glob
import os
import sys
import time
import logging
import itertools
import math

# try:
#     sys.path.append(glob.glob('/home/user/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
try:
    sys.path.append(glob.glob('/home/UnrealEngine_4.26/Engine/Binaries/Linux/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np

from carla import VehicleLightState as vls

from numpy import random

# ========================================================================================== #

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ========================================================================================== #

total_iterations = 405
reset_interval = 9
total_resets = math.ceil(total_iterations / reset_interval)
restart_iteration = 0

town = 'Town06'
# vehicles_list = []
client = carla.Client('127.0.0.1', 2000)
client.load_world(town)
client.set_timeout(60.0) # 10.0
print("[Info] town :", town)

for reset_index in range(total_resets):
    try:
        print("========== Reset ==========")

        world = client.get_world()
        world.unload_map_layer(carla.MapLayer.Buildings)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        #settings.no_rendering_mode = True
        world.apply_settings(settings)

        blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')

        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
        blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
        blueprints = [x for x in blueprints if not x.id.endswith('fusorosa')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        number_of_vehicles_options = [10, 30, 50] # [150, 200, 250]
        number_of_vehicles_cycle = itertools.cycle(number_of_vehicles_options)
        # for iteration in range(total_iterations - last_iteration):
        for index_btw_interval in range(reset_interval):
            print("========== Iteration", ((reset_index * reset_interval) + index_btw_interval + restart_iteration), "==========")
            timestamp_step_start = time.time()

            random.shuffle(spawn_points)
            vehicles_list = []
            batch = []
            number_of_vehicles = next(number_of_vehicles_cycle)
            print("[Info] number_of_vehicles :", number_of_vehicles)
            for n, transform in enumerate(spawn_points):
                if n >= number_of_vehicles:
                    break
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')

                # spawn the cars and set their autopilot and light state all together
                batch.append(SpawnActor(blueprint, transform)
                    .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

            for response in client.apply_batch_sync(batch, True):
                if response.error:
                    logging.error(response.error)
                else:
                    vehicles_list.append(response.actor_id)

            all_vehicle_actors = world.get_actors(vehicles_list) # carla.ActorList

            #for actor in all_vehicle_actors:
                #traffic_manager.ignore_lights_percentage(actor,25.0)
                #traffic_manager.ignore_vehicles_percentage(actor, 5.0)
                #traffic_manager.global_percentage_speed_difference(-30.0)
            
            world.tick()
            save_states = []
            for i in range(5000):
                state_vector = []
                for actor in all_vehicle_actors: # carla.Vehicle inherited from carla.Actor
                    r = random.random()
                    if r < 0.01:
                        #traffic_manager.force_lane_change(actor, True)
                        traffic_manager.distance_to_leading_vehicle(actor, 5)
                    #elif r < 0.02:
                        #traffic_manager.force_lane_change(actor, False)
                        #traffic_manager.global_lane_offset(20)
                    #elif r < 0.5:
                        #traffic_manager.force_lane_change(actor, True)
                        #traffic_manager.global_lane_offset(-20)

                    actor_transform = actor.get_transform() # carla.Transform -> carla.Location -> x, y: float meter    carla.Rotation -> pitch, yaw, roll: float -180~180degee
                    actor_velocity = actor.get_velocity() # carla.Vector3D -> x, y, z: float m/s
                    actor_angular_velocity = actor.get_angular_velocity() # carla.Vector3D -> x, y, z: deg/s
                    actor_acceleration = actor.get_acceleration() # carla.Vector3D -> x, y, z: m/s^2
                    actor_is_at_stop_line = float(actor.is_at_traffic_light()) # bool to float
                    traffic_light = actor.get_traffic_light_state() # carla.TrafficLightState, if in red red else green
                    if traffic_light == carla.TrafficLightState.Red:
                        actor_traffic_light_state = 0.0
                    elif traffic_light == carla.TrafficLightState.Yellow:
                        actor_traffic_light_state = 0.5
                    elif traffic_light == carla.TrafficLightState.Green:
                        actor_traffic_light_state = 1.0
                    elif traffic_light == carla.TrafficLightState.Off:
                        print("[Warn] TRAFFIC LIGHT IS OFF!!!")
                        exit()
                    else:
                        print("[Warn] TRAFFIC LIGHT IS UNKNOWN!!!")
                        exit()

                    state = [actor_transform.location.x, actor_transform.location.y, actor_transform.rotation.yaw, actor_velocity.x, actor_velocity.y, actor_angular_velocity.z, actor_acceleration.x, actor_acceleration.y, actor_is_at_stop_line, actor_traffic_light_state]
                    state_vector.append(state)

                world.tick()

                save_states.append(state_vector)

            save_states = np.array(save_states)

            np.save("gathered/log_speed_0percent/" + str((reset_index * reset_interval) + index_btw_interval + restart_iteration) + ".npy", save_states)
            
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

            timestamp_step_end = time.time()
            print(f"[Time] {timestamp_step_end - timestamp_step_start:.1f} seconds\n")

    finally:
        print("BLACK OUT")
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        time.sleep(5.0)