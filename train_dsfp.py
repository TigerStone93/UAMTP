"""
python: 3.8
nvidia driver: >= 450.80.02
torch: 1.13.1
CUDA: 11.7

sudo apt-get install lm-sensors
"""

import json
import numpy as np
import argparse
import re
import random
import math
import time
import torch
from torch import optim
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import gc
import os
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from resnet_dsfp import resnet18

import cv2
import matplotlib.pyplot as plt

import subprocess

import io
import threading

from torch.utils.tensorboard import SummaryWriter
import csv

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ========================================================================================== #

models = {"resnet18": resnet18,}

num_velocity_classes = 25
num_yaw_diff_classes = 71

# ========================================================================================== #

spf = 0.05

compensator = None
map_cropping_size = 320
scale = 6.0

traffic_light_radius = 3
traffic_light_padding = 1.5
spacing_from_vehicle = 7

vehicle_width = 2 * scale
vehicle_height = 5 * scale

# ========================================================================================== #

def training_args():
    learning_rate = 0.05
    momentum = 0.8 # 0.9 # higher value for accelerating convergence.
    optimizer = "sgd"
    loss = "cross_entropy"
    weight_decay = 5e-4 # 0.001 # 5e-4 # 0.0005 # higher value for bigger dataset and more complex model
    log_interval = 50
    pause_interval = 50
    save_interval = 50
    save_loc = "./"
    epoch = 300
    first_milestone = 90
    second_milestone = 170
    third_milestone = 240
    learning_rate_gamma = 0.5 # 0.1->0.05->0.025->0.0125 / 0.1->0.03->0.009->0.0027 / 0.1->0.02->0.004->0.0008 / 0.1->0.01->0.001->0.0001

    model = "resnet18"

    parser = argparse.ArgumentParser(description="Args for training parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")
    parser.add_argument("--dataset", type=str, dest="dataset", help="dataset to train on",)
    parser.add_argument("--pretrained", action="store_true", dest="pretrained", help="Load pretrained model")

    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Use GPU")
    parser.add_argument("--model", type=str, default=model, dest="model", help="Model to train")

    parser.add_argument("--epoch", type=int, default=epoch, dest="epoch", help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=learning_rate, dest="learning_rate", help="Learning rate",)
    parser.add_argument("--mom", type=float, default=momentum, dest="momentum", help="Momentum")
    parser.add_argument("--nesterov", action="store_true", dest="nesterov", help="Whether to use nesterov momentum in SGD",)
    parser.set_defaults(nesterov=False)
    parser.add_argument("--decay", type=float, default=weight_decay, dest="weight_decay", help="Weight Decay",)
    parser.add_argument("--opt", type=str, default=optimizer, dest="optimizer", help="Choice of optimisation algorithm",)

    parser.add_argument("--loss", type=str, default=loss, dest="loss_function", help="Loss function to be used for training",)

    parser.add_argument("--log-interval", type=int, default=log_interval, dest="log_interval", help="Log Interval on Terminal",)
    parser.add_argument("--pause-interval", type=int, default=pause_interval, dest="pause_interval", help="Pause Interval on Terminal",)
    parser.add_argument("--save-interval", type=int, default=save_interval, dest="save_interval", help="Save Interval on Terminal",)
    parser.add_argument("--save-path", type=str, default=save_loc, dest="save_loc", help="Path to export the model",)

    parser.add_argument("--first-milestone", type=int, default=first_milestone, dest="first_milestone", help="First milestone to change lr",)
    parser.add_argument("--second-milestone", type=int, default=second_milestone, dest="second_milestone", help="Second milestone to change lr",)
    parser.add_argument("--third-milestone", type=int, default=third_milestone, dest="third_milestone", help="Second milestone to change lr",)
    parser.add_argument("--learning-rate-gamma", type=float, default=learning_rate_gamma, dest="learning_rate_gamma", help="Gamma to change lr",)

    return parser

# ========================================================================================== #

def draw_wedge_patterned_line(img, pt1, pt2, color, ws=7):
    wedge_size = ws
    gap_size = 0
    thickness = 4

    length = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if length < 1e-4:  # Consider a small threshold for drawing very short lines
        cv2.line(img, tuple(pt1), tuple(pt2), color, thickness)
        return
    num_wedges = max(int(length / (wedge_size + gap_size)), 1)  # Ensure at least one wedge

    # Unit direction vector
    ux, uy = (pt2[0] - pt1[0]) / length, (pt2[1] - pt1[1]) / length

    # Perpendicular unit vector
    px, py = -uy, ux

    for i in range(num_wedges):
        start = np.array([pt1[0] + i * (wedge_size + gap_size) * ux, pt1[1] + i * (wedge_size + gap_size) * uy])
        end = start + np.array([wedge_size * ux, wedge_size * uy])
        
        # Convert points to integer type before passing to cv2 functions
        A = tuple((end).astype(int))
        B = tuple((start + np.array([px, py]) * (wedge_size / 2)).astype(int))
        C = tuple((start - np.array([px, py]) * (wedge_size / 2)).astype(int))

        # Check if points are inside the image bounds before drawing
        if all(0 <= x < img.shape[1] and 0 <= y < img.shape[0] for x, y in [A, B, C]):
            cv2.fillConvexPoly(img, np.array([A, B, C]), color)
        else:
            break # Break if a wedge is out of bounds

# ========================================================================================== #

def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

# ========================================================================================== #

def get_gpu_temperatures():
    try:
        result = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader", shell=True)
        temperatures = result.decode('utf-8').strip().split('\n')
        return temperatures
    except subprocess.CalledProcessError as e:
        print("[Warn] Failed to execute nvidia-smi:", e)
        return []

# ========================================================================================== #

def get_cpu_temperatures():
    process = subprocess.Popen(['sensors'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decode the byte string
    output = stdout.decode('utf-8')

    # Find lines containing 'Core' to extract CPU temperatures
    lines = output.split('\n')
    temperatures = []
    for line in lines:
        if "Core" in line:
            temp_str = line.split(':')[1].split('(')[0].strip()
            temperatures.append(temp_str)
    return temperatures
    
# ========================================================================================== #

def async_save_model(model, optimizer, save_model_path, save_optimizer_path):
    try:
        torch.save(model.module.state_dict(), save_model_path)
        print("[Info] save_model_path :", save_model_path)

        torch.save(optimizer.state_dict(), save_optimizer_path)
        print("[Info] save_optimizer_path :", save_optimizer_path)
    except Exception as e:
        print(f"[Warn] Error saving model or optimizer: {e}")

# ========================================================================================== #
# ========================================================================================== #
# ========================================================================================== #

if __name__ == "__main__":
    # Parsing the arguments
    args = training_args().parse_args()
    
    # ============================== #
    
    # Setting the seed
    print("[Info] args :", args)
    print("[Info] seed :", args.seed)
    torch.manual_seed(args.seed)

    # Setting the device
    cuda = torch.cuda.is_available() and args.gpu
    if cuda:
        print(f"[Info] PyTorch CUDA version : {torch.version.cuda}")
        accelerator = Accelerator()
        device = accelerator.device
    else:
        device = torch.device("cpu")
    print("[Info] CUDA :", str(cuda))
    
    # ============================== #
    
    # Choosing the model to train
    net = models[args.model]()
    
    # ============================== #
    
    start_epoch_index = 0
    if args.pretrained:
        load_path = "./resnet18_0_log_speed_0percent_150_model.pth"
        print("[Info] load_path :", load_path)
        checkpoint = torch.load(load_path)
        net.load_state_dict(checkpoint)

        pattern = r"(\d+)_model\.pth$"
        match = re.search(pattern, load_path)
        start_epoch_index = int(match.group(1)) if match else 0

        if start_epoch_index > args.first_milestone:
            args.learning_rate *= args.learning_rate_gamma
        if start_epoch_index > args.second_milestone:
            args.learning_rate *= args.learning_rate_gamma
        if start_epoch_index > args.third_milestone:
            args.learning_rate *= args.learning_rate_gamma

    print("[Info] start_epoch_index :", start_epoch_index)
    print("[Info] learning_rate :", args.learning_rate)

    # ============================== #
    
    # Using the gpu
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("[Info] device_count :", torch.cuda.device_count())
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = False # True
        torch.backends.cudnn.deterministic = False # Default
    
    # ============================== #
    
    # Choosing the optimizer
    opt_params = net.parameters()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(opt_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov,)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    for param_group in optimizer.param_groups:
        param_group.setdefault('initial_lr', args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.first_milestone, args.second_milestone, args.third_milestone], gamma=args.learning_rate_gamma, last_epoch=start_epoch_index-1)
    print("[Info] scheduler.last_epoch :", scheduler.last_epoch)

    # ============================== #
    
    # Drawing the base map image for visualizing
    map_lanes_path = "data/lane_town03.txt"
    print("[Info] map_lanes_path :", map_lanes_path)

    lanes = []
    with open(map_lanes_path, "rt") as rf:
        for line in rf.readlines():
            lane = []
            for s in line.split("\t"):
                v = s.split(",")
                if len(v) == 2:
                    lane.append([float(v[0]), float(v[1])])
            if len(lane) > 0:
                lanes.append(np.array(lane))

    if "town03" in map_lanes_path:
        world_map = np.full((4100, 4000, 3), 128, np.uint8)
        compensator = np.array([200, 256])

    elif "town05" in map_lanes_path:
        world_map_input = np.full((4500, 5000, 3), 128, np.uint8)
        compensator = np.array([350, 280])

    elif "town06" in map_lanes_path:
        world_map_input = np.full((5700, 9500, 3), 128, np.uint8)
        compensator = np.array([440, 230])

    elif "town07" in map_lanes_path:
        world_map = np.full((4100, 3500, 3), 128, np.uint8)
        compensator = np.array([280, 320])

    for lane in lanes:
        for i, _ in enumerate(lane[:-1]):
            dx = lane[i+1][0] - lane[i][0]
            dy = lane[i+1][1] - lane[i][1]
            r = np.sqrt(dx * dx + dy * dy)
            if r > 0.1:
                color = ( int(dx * 127 / r + 128), 128, int(dy * 127 / r + 128) )
                start_point = ((lane[i] + compensator) * scale).astype(np.int32)
                end_point = ((lane[i+1] + compensator) * scale).astype(np.int32)
                draw_wedge_patterned_line(world_map, start_point, end_point, color, 7)

    # ============================== #
    
    # Creating the summary writer in tensorboard
    writer = SummaryWriter(args.save_loc + "stats_logging/")
    training_set_loss = {}
    save_name = str(args.model) + "_seed_" + str(args.seed)
    print("[Info] save_name :", save_name)
    
    dataset_name = "log_speed_0percent_town03_aug_150200250"
    dataset_directory = "data" + "/" + dataset_name + "/"
    print("[Info] dataset_directory :", dataset_directory)
    number_of_dataset = 501
    
    # ============================== #
    
    # csv_file = open('epoch_all_acc_angvel.csv', 'w')
    # csv_writer = csv.writer(csv_file)

    # csv_writer.writerow(["Current Acceleration", "After 10s Acceleration", "After 20s Acceleration", "After 30s Acceleration", "After 40s Acceleration",
    #                      "Current Angular Velocity", "After 10s Angular Velocity", "After 20s Angular Velocity", "After 30s Angular Velocity", "After 40s Angular Velocity"])

    # ============================== #

    net.train()
    for epoch in range(start_epoch_index, args.epoch):
        print("========== Epoch", epoch, "==========")
        timestamp_step_start = time.time()
        
        # Loading the matrix dataset for preprocessing
        for attempt in range(5):
            try:
                random_file_index = random.randrange(number_of_dataset)
                record = np.load(dataset_directory + str(random_file_index) + ".npy")
                record_index_shuffled = list(range(10, np.shape(record)[0] - 40)) # Discarding the first 10 indices and last 40 indices
                random.shuffle(record_index_shuffled)
                break
            except Exception as e:
                print(f"[Warn] Failed to load a file {dataset_directory}{random_file_index}.npy. Error: {e}")
                continue
        else:
            print("[Warn] Failed to load a file after 5 attempts.")
            exit()
        
        # ============================== #
        
        # Sampling the 100 indices from 0 to 4970
        num_index_samples = 70 # 50
        num_vehicle_samples = 125 # Vehicles are spawned in random points for each iteration.
        training_epoch_loss = 0
        # whole_counter = 0
        # vanishing_counter = 0
        # stationary_counter = 0
        # angular_velocity_counter = 0
        for step in record_index_shuffled[:num_index_samples]:
            before_10_record = record[step-10]
            before_5_record = record[step-5]
            current_record = record[step] # (total_vehicles, [location.x, locataion.y, rotation.yaw, v.x, v.y, angvel, acc.x, acc.y, is_at_stop_line, traffic_light])

            before_10_xy_all = before_10_record[:, 0:2]
            before_5_xy_all = before_5_record[:, 0:2]
            current_xy_all = current_record[:, 0:2]

            # current_is_at_stop_line = current_record[:, 8] # (total_vehicles,) not (total_vehicles, 1)
            
            before_10_xy = before_10_record[:num_vehicle_samples, 0:2]

            current_xy = current_record[:num_vehicle_samples, 0:2] # meter
            current_yaw = current_record[:num_vehicle_samples, 2:3] # -180~180 degree, positive yaw: counterclockwise, negative yaw: clockwise
            current_velocity_xy = current_record[:num_vehicle_samples, 3:5] # m/s
            current_angular_velocity = current_record[:num_vehicle_samples, 5:6] # deg/s
            current_acceleration_xy = current_record[:num_vehicle_samples, 6:8] # m/s^2
            # current_intersection_info = current_record[:num_vehicle_samples, 8:10]

            after_10_xy = record[step+10, :num_vehicle_samples, 0:2]
            after_10_yaw = record[step+10, :num_vehicle_samples, 2:3]
            after_10_velocity_xy = record[step+10, :num_vehicle_samples, 3:5]
            after_10_angular_velocity = record[step+10, :num_vehicle_samples, 5:6]
            after_10_acceleration_xy = record[step+10, :num_vehicle_samples, 6:8]

            after_20_xy = record[step+20, :num_vehicle_samples, 0:2]
            after_20_yaw = record[step+20, :num_vehicle_samples, 2:3] 
            after_20_velocity_xy = record[step+20, :num_vehicle_samples, 3:5]
            after_20_angular_velocity = record[step+20, :num_vehicle_samples, 5:6]
            after_20_acceleration_xy = record[step+20, :num_vehicle_samples, 6:8]

            after_30_xy = record[step+30, :num_vehicle_samples, 0:2]
            after_30_yaw = record[step+30, :num_vehicle_samples, 2:3] 
            after_30_velocity_xy = record[step+30, :num_vehicle_samples, 3:5]
            after_30_angular_velocity = record[step+30, :num_vehicle_samples, 5:6]
            after_30_acceleration_xy = record[step+30, :num_vehicle_samples, 6:8]
            
            after_40_xy = record[step+40, :num_vehicle_samples, 0:2]
            after_40_yaw = record[step+40, :num_vehicle_samples, 2:3]
            after_40_velocity_xy = record[step+40, :num_vehicle_samples, 3:5]
            after_40_angular_velocity = record[step+40, :num_vehicle_samples, 5:6]
            after_40_acceleration_xy = record[step+40, :num_vehicle_samples, 6:8]

            # Sampling for making labels
            combined_record_sampled = np.concatenate((before_10_xy,
                                                      current_xy, current_yaw, current_velocity_xy, current_angular_velocity, current_acceleration_xy,
                                                      after_10_xy, after_10_yaw, after_10_velocity_xy, after_10_angular_velocity, after_10_acceleration_xy,
                                                      after_20_xy, after_20_yaw, after_20_velocity_xy, after_20_angular_velocity, after_20_acceleration_xy,
                                                      after_30_xy, after_30_yaw, after_30_velocity_xy, after_30_angular_velocity, after_30_acceleration_xy,
                                                      after_40_xy, after_40_yaw, after_40_velocity_xy, after_40_angular_velocity, after_40_acceleration_xy), axis=1)
            
            # ============================== #

            # Generating the grid labels by preprocessing
            label_after_10_velocity_list = []
            label_after_20_velocity_list = []
            label_after_30_velocity_list = []
            label_after_40_velocity_list = []
            label_after_10_yaw_diff_list = []
            label_after_20_yaw_diff_list = []
            label_after_30_yaw_diff_list = []
            label_after_40_yaw_diff_list = []
            counter_exclude = 0
            counter_include_stationary = 0
            counter_include_angular_velocity = 0
            counter_exclude_list = []
            counter_visualize = 0

            for cr in combined_record_sampled:
                # whole_counter += 1
                before_10_x, before_10_y, \
                current_x, current_y, current_yaw, current_velocity_x, current_velocity_y, current_angular_velocity, current_acceleration_x, current_acceleration_y, \
                after_10_x, after_10_y, after_10_yaw, after_10_velocity_x, after_10_velocity_y, after_10_angular_velocity, after_10_acceleration_x, after_10_acceleration_y, \
                after_20_x, after_20_y, after_20_yaw, after_20_velocity_x, after_20_velocity_y, after_20_angular_velocity, after_20_acceleration_x, after_20_acceleration_y, \
                after_30_x, after_30_y, after_30_yaw, after_30_velocity_x, after_30_velocity_y, after_30_angular_velocity, after_30_acceleration_x, after_30_acceleration_y, \
                after_40_x, after_40_y, after_40_yaw, after_40_velocity_x, after_40_velocity_y, after_40_angular_velocity, after_40_acceleration_x, after_40_acceleration_y = cr

                current_velocity = math.sqrt(current_velocity_x**2 + current_velocity_y**2) # * 3.6
                after_10_velocity = math.sqrt(after_10_velocity_x**2 + after_10_velocity_y**2)
                after_20_velocity = math.sqrt(after_20_velocity_x**2 + after_20_velocity_y**2)
                after_30_velocity = math.sqrt(after_30_velocity_x**2 + after_30_velocity_y**2)
                after_40_velocity = math.sqrt(after_40_velocity_x**2 + after_40_velocity_y**2)

                current_acceleration_abs = math.sqrt(current_acceleration_x**2 + current_acceleration_y**2)
                after_10_acceleration_abs = math.sqrt(after_10_acceleration_x**2 + after_10_acceleration_y**2)
                after_20_acceleration_abs = math.sqrt(after_20_acceleration_x**2 + after_20_acceleration_y**2)
                after_30_acceleration_abs = math.sqrt(after_30_acceleration_x**2 + after_30_acceleration_y**2)
                after_40_acceleration_abs = math.sqrt(after_40_acceleration_x**2 + after_40_acceleration_y**2)
                
                # ============================== #
                
                # Filtering out data with vanishing bug (almost 0% of data)
                if current_x == current_y == 0 or after_10_x == after_10_y == 0 or after_20_x == after_20_y == 0 or after_30_x == after_30_y == 0 or after_40_x == after_40_y == 0:
                    # vanishing_counter += 1
                    counter_exclude_list.append(counter_exclude)
                    counter_exclude += 1
                    continue
                    
                # ============================== #
                
                # Filtering out some data with stationary state (almost 50% of data)
                if before_10_x == current_x == after_10_x == after_20_x == after_30_x == after_40_x and before_10_y == current_y == after_10_y == after_20_y == after_30_y == after_40_y:
                    # stationary_counter += 1
                    if counter_include_stationary % 10 == 0:
                        counter_include_stationary += 1
                    else:
                        counter_exclude_list.append(counter_exclude)
                        counter_exclude += 1
                        counter_include_stationary += 1
                        continue
                    
                # ============================== #

                # Filtering out data with invalid speed
                if not ((0 <= current_velocity * 3.6 <= 90) or (0 <= after_10_velocity * 3.6 <= 90) or (0 <= after_20_velocity * 3.6 <= 90) or (0 <= after_30_velocity * 3.6 <= 90) or (0 <= after_40_velocity * 3.6 <= 90)):
                    counter_exclude_list.append(counter_exclude)
                    counter_exclude += 1
                    print(f"Velocity currently : ({current_velocity:.2f})")
                    print(f"Velocity after 10 timestep : ({after_10_velocity:.2f})")
                    print(f"Velocity after 20 timestep : ({after_20_velocity:.2f})")
                    print(f"Velocity after 30 timestep : ({after_30_velocity:.2f})")
                    print(f"Velocity after 40 timestep : ({after_40_velocity:.2f})")
                    continue
                    
                # ============================== #

                # velocity label: 0~24 (25 classes) (range: 1m/s)
                # ex) 0~1m/s: 0, 1~2m/s: 1, 2~3m/s: 2
                velocity_label_bin_width = 1.0
                label_after_10_velocity = math.floor(after_10_velocity / velocity_label_bin_width)
                label_after_20_velocity = math.floor(after_20_velocity / velocity_label_bin_width)
                label_after_30_velocity = math.floor(after_30_velocity / velocity_label_bin_width)
                label_after_40_velocity = math.floor(after_40_velocity / velocity_label_bin_width)

                # yaw difference label: -35~35 (71 classes) (range: 5deg/s)
                # ex) -177.5~-172.5deg/s: -35, ... ,-12.5~-7.5deg/s: -2, -7.5~-2.5deg/s: -1, -2.5~2.5deg/s: 0, 2.5~7.5deg/s: 1, 7.5~12.5deg/s: 2, ... , 172.5~177.5deg/s: 35
                yaw_diff_label_bin_width = 5.0
                after_10_yaw_diff = after_10_yaw - current_yaw
                if after_10_yaw_diff > 180: after_10_yaw_diff -= 360
                elif after_10_yaw_diff < -180: after_10_yaw_diff += 360
                after_20_yaw_diff = after_20_yaw - current_yaw
                if after_20_yaw_diff > 180: after_20_yaw_diff -= 360
                elif after_20_yaw_diff < -180: after_20_yaw_diff += 360
                after_30_yaw_diff = after_30_yaw - current_yaw
                if after_30_yaw_diff > 180: after_30_yaw_diff -= 360
                elif after_30_yaw_diff < -180: after_30_yaw_diff += 360
                after_40_yaw_diff = after_40_yaw - current_yaw
                if after_40_yaw_diff > 180: after_40_yaw_diff -= 360
                elif after_40_yaw_diff < -180: after_40_yaw_diff += 360

                label_after_10_yaw_diff = math.floor((after_10_yaw_diff + yaw_diff_label_bin_width/2) / yaw_diff_label_bin_width)
                label_after_10_yaw_diff += 35
                label_after_20_yaw_diff = math.floor((after_20_yaw_diff + yaw_diff_label_bin_width/2) / yaw_diff_label_bin_width)
                label_after_20_yaw_diff += 35
                label_after_30_yaw_diff = math.floor((after_30_yaw_diff + yaw_diff_label_bin_width/2) / yaw_diff_label_bin_width)
                label_after_30_yaw_diff += 35
                label_after_40_yaw_diff = math.floor((after_40_yaw_diff + yaw_diff_label_bin_width/2) / yaw_diff_label_bin_width)
                label_after_40_yaw_diff += 35

                label_after_10_velocity_list.append(label_after_10_velocity)
                label_after_20_velocity_list.append(label_after_20_velocity)
                label_after_30_velocity_list.append(label_after_30_velocity)
                label_after_40_velocity_list.append(label_after_40_velocity)

                label_after_10_yaw_diff_list.append(label_after_10_yaw_diff)
                label_after_20_yaw_diff_list.append(label_after_20_yaw_diff)
                label_after_30_yaw_diff_list.append(label_after_30_yaw_diff)
                label_after_40_yaw_diff_list.append(label_after_40_yaw_diff)
                
                # ============================== #
                
                # Increasing the number of counter
                counter_exclude += 1
                counter_visualize += 1

                # ============================== #

                # csv_writer.writerow([current_acceleration, after_10_acceleration, after_20_acceleration, after_30_acceleration, after_40_acceleration,
                #                      current_angular_velocity, after_10_angular_velocity, after_20_angular_velocity, after_30_angular_velocity, after_40_angular_velocity])

            # ============================== #
            
            # print("whole_counter :", whole_counter)
            # print("vanishing_counter :", vanishing_counter)
            # print("stationary_counter :", stationary_counter)
            # print("angular_velocity_counter :", angular_velocity_counter)
            # print(counter_exclude_list)
            
            # ============================== #
            
            # Filtering out the record data according to the conditions above
            current_record_sampled_filtered = np.delete(current_record[:num_vehicle_samples], counter_exclude_list, axis=0) # (num_vehicle_samples, 10)
            history_record_concat = np.concatenate([current_record[:num_vehicle_samples], before_5_record[:num_vehicle_samples], before_10_record[:num_vehicle_samples]], axis=1)
            history_record_sampled_filtered = np.delete(history_record_concat, counter_exclude_list, axis=0) # (num_vehicle_samples - n, 30)

            # ============================== #
            
            # Generating the map inputs by preprocessing
            world_map_input_copied = world_map.copy()

            # Drawing the rectangles representing location of vehicles on map for all vehicles, including unsampled ones
            for cr in current_record:
                location = tuple(((np.array(cr[:2]) + compensator) * scale).astype(int))
                yaw_radian = np.radians(cr[2] + 90)
                points = np.array([[-vehicle_width/2, -vehicle_height/2],
                                   [vehicle_width/2, -vehicle_height/2],
                                   [vehicle_width/2, vehicle_height/2],
                                   [-vehicle_width/2, vehicle_height/2]])
                rotated_points = np.array([rotate_point((0,0), point, yaw_radian) for point in points])
                converted_points = (rotated_points + [location[0], location[1]]).astype(np.float32)

                rectangle = cv2.minAreaRect(converted_points)
                box = cv2.boxPoints(rectangle)
                box = np.int0(box)
                cv2.drawContours(world_map_input_copied, [box], 0, (0, 0, 0), -1)

            # ============================== #

            # Drawing the trajectory history of vehicles on map for all vehicles, including unsampled ones
            for i in range(len(current_record)):
                # Convert locations to map coordinates
                before_10_location = tuple(((np.array(before_10_xy_all[i]) + compensator) * scale).astype(int))
                before_5_location = tuple(((np.array(before_5_xy_all[i]) + compensator) * scale).astype(int))
                current_location = tuple(((np.array(current_xy_all[i]) + compensator) * scale).astype(int))

                # Draw lines for the trajectory
                cv2.line(world_map_input_copied, before_10_location, before_5_location, (0, 0, 255), 3) # 4 or int(0.5 * scale)
                cv2.line(world_map_input_copied, before_5_location, current_location, (0, 0, 255), 3) # 4 or int(0.5 * scale)
 
            # ============================== #
                           
            # Drawing the affected traffic light on map for all vehicles, including unsampled ones
            # TO MAKE THE TRAFFIC LIGHT APPEAR ABOVE THE CAR, DO NOT COMBINE THE PROCESS OF DRAWING THE VEHICLE WITH THE PROCESS OF DRAWING THE TRAFFIC LIGHT.
            for cr in current_record:
                if cr[8] == 1.0: # current_is_at_stop_line
                    initial_location = tuple(((np.array(cr[:2]) + compensator) * scale).astype(int))
                    initial_yaw_radian = np.radians(cr[2] + 90)

                    # Calculating the total height and width of the traffic light (3 circles vertically with padding)
                    traffic_light_total_height = traffic_light_radius * 6 + traffic_light_padding * 4
                    traffic_light_total_width = traffic_light_radius * 2 + traffic_light_padding * 2

                    # Calculating the offset from the vehicle's center to the right side where the traffic light will be drawn
                    traffic_light_offset_x = vehicle_width / 2 + spacing_from_vehicle
                    traffic_light_offset_y = -traffic_light_total_height / 2

                    gray_rect_top_left_before_rotation = (traffic_light_offset_x - traffic_light_radius - traffic_light_padding,
                                                          -traffic_light_total_height // 2)
                    gray_rect_bottom_right_before_rotation = (gray_rect_top_left_before_rotation[0] + traffic_light_radius * 2 + traffic_light_padding * 2,
                                                              gray_rect_top_left_before_rotation[1] + traffic_light_total_height)
                    
                    # Rotating the corners of the rectangle
                    gray_rect_top_left = rotate_point((0, 0), gray_rect_top_left_before_rotation, initial_yaw_radian)
                    gray_rect_bottom_right = rotate_point((0, 0), gray_rect_bottom_right_before_rotation, initial_yaw_radian)

                    # Translating the rotated corners to the vehicle's position on the map
                    gray_rect_top_left = (int(initial_location[0] + gray_rect_top_left[0]),
                                          int(initial_location[1] + gray_rect_top_left[1]))
                    gray_rect_bottom_right = (int(initial_location[0] + gray_rect_bottom_right[0]),
                                              int(initial_location[1] + gray_rect_bottom_right[1]))
                    
                    # Drawing the gray rectangle for traffic light background
                    cv2.rectangle(world_map_input_copied, gray_rect_top_left, gray_rect_bottom_right, (128, 128, 128), -1)

                    # Calculating and rotating the positions of the traffic light circles before translation
                    circle_positions_before_rotation = [(traffic_light_offset_x, -traffic_light_total_height // 2 + traffic_light_padding + traffic_light_radius),
                                                        (traffic_light_offset_x, -traffic_light_total_height // 2 + 2 * traffic_light_padding + 3 * traffic_light_radius),
                                                        (traffic_light_offset_x, -traffic_light_total_height // 2 + 3 * traffic_light_padding + 5 * traffic_light_radius)]

                    # Rotating and translating circle positions
                    circle_positions = [rotate_point((0, 0), pos, initial_yaw_radian) for pos in circle_positions_before_rotation]
                    circle_positions_on_map = [(int(initial_location[0] + pos[0]), int(initial_location[1] + pos[1])) for pos in circle_positions]

                    # Drawing the traffic light circles using the rotated and translated positions
                    for i, pos in enumerate(circle_positions_on_map):
                        color = (0, 0, 0) # Default to off/black
                        if i == 0 and cr[9] == 0.0: # Red light
                            color = (0, 0, 255)
                        elif i == 1 and cr[9] == 0.5: # Yellow light
                            color = (0, 255, 255)
                        elif i == 2 and cr[9] == 1.0: # Green light
                            color = (0, 255, 0)
                        cv2.circle(world_map_input_copied, pos, traffic_light_radius, color, -1)

            # ============================== #
                    
            map_input_list = []
            for cr in current_record_sampled_filtered:
                location = (cr[:2] + compensator) * scale
                yaw = cr[2] + 90
                M1 = np.float32( [ [1, 0, -location[0]], [0, 1, -location[1]], [0, 0, 1] ] )
                M2 = cv2.getRotationMatrix2D((0, 0), yaw, 1.0)
                M2 = np.append(M2, np.float32([[0, 0, 1]]), axis=0)
                M3 = np.float32( [ [1, 0, map_cropping_size/2], [0, 1, map_cropping_size*3/4], [0, 0, 1] ] )
                M = np.matmul(np.matmul(M3, M2), M1)
                map_rotated_n_cropped = cv2.warpAffine(world_map_input_copied, M[:2], (map_cropping_size, map_cropping_size)) # (width, height)
                map_input_list.append(map_rotated_n_cropped.astype(np.float32) / 128.0 - 1.0) # (num_vehicle_samples, map_cropping_size, map_cropping_size, 3)

                # ============================== #
                
                # Visualizing the map
                """
                map_rotated_n_cropped = cv2.cvtColor(map_rotated_n_cropped, cv2.COLOR_BGR2RGB)
                plt.imshow(map_rotated_n_cropped)
                plt.axis('off')
                plt.show()
                """

            # print("map_input_list shape :", np.array(map_input_list).shape) # (number_of_vehicles, map_cropping_size height, map_cropping_size width, 3)

            # ============================== #

            # Converting the arrays to tensors for inputs of model
            map_input_tensor = (torch.tensor(np.array(map_input_list), dtype=torch.float32, requires_grad=True).permute(0, 3, 1, 2)).to(device) # (num_vehicle_samples - len(counter_exclude_list), map_cropping_size height, map_cropping_size width, 3 channels) → (num_vehicle_samples - len(counter_exclude_list), 3 channels, map_cropping_size height, map_cropping_size width)
            record_input_tensor = torch.tensor(history_record_sampled_filtered, dtype=torch.float32, requires_grad=True).to(device)
            label_after_10_velocity_tensor = torch.tensor(np.array(label_after_10_velocity_list), requires_grad=False).to(device)
            label_after_20_velocity_tensor = torch.tensor(np.array(label_after_20_velocity_list), requires_grad=False).to(device)
            label_after_30_velocity_tensor = torch.tensor(np.array(label_after_30_velocity_list), requires_grad=False).to(device)
            label_after_40_velocity_tensor = torch.tensor(np.array(label_after_40_velocity_list), requires_grad=False).to(device)
            label_after_10_yaw_diff_tensor = torch.tensor(np.array(label_after_10_yaw_diff_list), requires_grad=False).to(device)
            label_after_20_yaw_diff_tensor = torch.tensor(np.array(label_after_20_yaw_diff_list), requires_grad=False).to(device) 
            label_after_30_yaw_diff_tensor = torch.tensor(np.array(label_after_30_yaw_diff_list), requires_grad=False).to(device)
            label_after_40_yaw_diff_tensor = torch.tensor(np.array(label_after_40_yaw_diff_list), requires_grad=False).to(device)
            
            # ============================== #

            # Getting the output by putting input to model
            optimizer.zero_grad()
            logit_after_10_velocity_tensor, logit_after_20_velocity_tensor, logit_after_30_velocity_tensor, logit_after_40_velocity_tensor, logit_after_10_yaw_diff_tensor, logit_after_20_yaw_diff_tensor, logit_after_30_yaw_diff_tensor, logit_after_40_yaw_diff_tensor = net(map_input_tensor, record_input_tensor)

            # ============================== #
            
            # Calculating the cross entropy loss by applying the softmax output
            loss_function_dict = {"cross_entropy": F.cross_entropy}
            cross_entropy_loss_velocity_1 = loss_function_dict[args.loss_function](logit_after_10_velocity_tensor, label_after_10_velocity_tensor) # 0 ~ inf
            cross_entropy_loss_velocity_2 = loss_function_dict[args.loss_function](logit_after_20_velocity_tensor, label_after_20_velocity_tensor) # 0 ~ inf
            cross_entropy_loss_velocity_3 = loss_function_dict[args.loss_function](logit_after_30_velocity_tensor, label_after_30_velocity_tensor) # 0 ~ inf
            cross_entropy_loss_velocity_4 = loss_function_dict[args.loss_function](logit_after_40_velocity_tensor, label_after_40_velocity_tensor) # 0 ~ inf
            cross_entropy_loss_yaw_diff_1 = loss_function_dict[args.loss_function](logit_after_10_yaw_diff_tensor, label_after_10_yaw_diff_tensor) # 0 ~ inf
            cross_entropy_loss_yaw_diff_2 = loss_function_dict[args.loss_function](logit_after_20_yaw_diff_tensor, label_after_20_yaw_diff_tensor) # 0 ~ inf
            cross_entropy_loss_yaw_diff_3 = loss_function_dict[args.loss_function](logit_after_30_yaw_diff_tensor, label_after_30_yaw_diff_tensor) # 0 ~ inf
            cross_entropy_loss_yaw_diff_4 = loss_function_dict[args.loss_function](logit_after_40_yaw_diff_tensor, label_after_40_yaw_diff_tensor) # 0 ~ inf

            # ============================== #

            # Calculating the loss for step which using num_vehicle_samples as batch
            cross_entropy_loss_velocity = 1/10*cross_entropy_loss_velocity_1 + 1/10*cross_entropy_loss_velocity_2 + 1/10*cross_entropy_loss_velocity_3 + 1/10*cross_entropy_loss_velocity_4
            cross_entropy_loss_yaw_diff = 1/10*cross_entropy_loss_yaw_diff_1 + 1/10*cross_entropy_loss_yaw_diff_2 + 1/10*cross_entropy_loss_yaw_diff_3 + 1/10*cross_entropy_loss_yaw_diff_4
            training_step_loss = cross_entropy_loss_velocity + cross_entropy_loss_yaw_diff
            
            # ============================== #

            # Calculating the gradient by backpropagation
            training_step_loss.backward()
            training_epoch_loss += training_step_loss.item()
            
            # ============================== #
            
            # Updating the parameters by gradient descent
            optimizer.step()

        # ============================== #

        # Calculating the loss for epoch which using num_index_samples as batch
        training_epoch_loss /= num_index_samples
        print(f"[Term] CE_V1 : {cross_entropy_loss_velocity_1:.4f},    CE_V2 : {cross_entropy_loss_velocity_2:.4f},    CE_V3 : {cross_entropy_loss_velocity_3:.4f},    CE_V4 : {cross_entropy_loss_velocity_4:.4f},    CE_Y1 : {cross_entropy_loss_yaw_diff_1:.4f},    CE_Y2 : {cross_entropy_loss_yaw_diff_2:.4f},    CE_Y3 : {cross_entropy_loss_yaw_diff_3:.4f},    CE_Y4 : {cross_entropy_loss_yaw_diff_4:.4f}    (Loss of very last record)")
        print(f"[Loss] {training_epoch_loss:.4f}")
        writer.add_scalar(save_name + "_training_epoch_loss", training_epoch_loss, (epoch + 1))
        training_set_loss[epoch] = training_epoch_loss
        
        print(f"[Learning Rate] {optimizer.param_groups[0]['lr']}")

        # ============================== #
        
        # Decaying the learning_rate according to milestones
        scheduler.step()

        # ============================== #

        # # Controlling the temperature of gpus
        if (epoch + 1) % args.pause_interval == 0:
            print("")
            for i, temperature in enumerate(get_gpu_temperatures()):
                print(f"[Info] GPU {i} Temperature : {temperature}°C")
            print("")
            for i, temperature in enumerate(get_cpu_temperatures()):
                print(f"[Info] CPU {i} Temperature : {temperature}°C")
            print("")
            pause_duration = 180 # seconds
            print(f"[Halt] Pausing for {pause_duration} seconds for heat control...")
            time.sleep(pause_duration)

            print("")
            for i, temperature in enumerate(get_gpu_temperatures()):
                print(f"[Info] GPU {i} Temperature : {temperature}°C")
            print("")
            for i, temperature in enumerate(get_cpu_temperatures()):
                print(f"[Info] CPU {i} Temperature : {temperature}°C")
            print("")
        
        # ============================== #

        # Saving the model per save_interval
        try:
            if (epoch + 1) % args.save_interval == 0:
                save_model_path = args.save_loc + save_name + "_" + dataset_name + "_" + str(epoch + 1) + "_model.pth"
                if cuda:
                    torch.save(net.module.state_dict(), save_model_path)
                else:
                    torch.save(net.state_dict(), save_model_path)
                print("[Info] save_model_path :", save_model_path)
        except Exception as e:
            print(f"[Warn] Failed to save a model. Error: {e}")
            exit()
        
        # ============================== #
          
        timestamp_step_end = time.time()
        print(f"[Time] {timestamp_step_end - timestamp_step_start:.1f} seconds\n")

    # ============================== #
        
    with open(save_model_path[: save_model_path.rfind("_")] + "_training_set_loss.json", "a") as f:
        json.dump(training_set_loss, f)
    
    # csv_file.close()