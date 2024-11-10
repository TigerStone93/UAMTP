import os
import json
import numpy as np
import random
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
import argparse
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import gc

from resnet_dsfp import resnet18

from collections import defaultdict

from sklearn.metrics import accuracy_score
from utils.temperature_scaling import ModelWithTemperature
from sklearn import metrics

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import csv
import pickle

# ========================================================================================== #

models = {"resnet18": resnet18,}

num_velocity_classes = 25
num_yaw_diff_classes = 71

model_to_num_dim = {"resnet18": 128} # in_feature size of final fully connected layer (classifier)

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

def evaluation_args():
    model = "resnet18"
    runs = 90 # 100

    parser = argparse.ArgumentParser(description="Training for calibration.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Use GPU")
    parser.set_defaults(gpu=False) # True
    parser.add_argument("--model", type=str, default=model, dest="model", help="Model to train")
    parser.add_argument("--runs", type=int, default=runs, dest="runs", help="Number of models to aggregate over",)

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

class Dataset(Dataset):
    def __init__(self, input_1, input_2, label_vel_1, label_vel_2, label_vel_3, label_vel_4, label_ang_1, label_ang_2, label_ang_3, label_ang_4):
        self.input_1 = input_1
        self.input_2 = input_2
        self.label_vel_1 = label_vel_1
        self.label_vel_2 = label_vel_2
        self.label_vel_3 = label_vel_3
        self.label_vel_4 = label_vel_4
        self.label_ang_1 = label_ang_1
        self.label_ang_2 = label_ang_2
        self.label_ang_3 = label_ang_3
        self.label_ang_4 = label_ang_4
        
    def __len__(self):
        return len(self.input_1)
        
    def __getitem__(self, index):
        return (self.input_1[index], self.input_2[index]), (self.label_vel_1[index], self.label_vel_2[index], self.label_vel_3[index], self.label_vel_4[index], self.label_ang_1[index], self.label_ang_2[index], self.label_ang_3[index], self.label_ang_4[index])

# ========================================================================================== #

def get_dataset(map_input_tensor, record_input_tensor,
                label_after_10_velocity_np, label_after_20_velocity_np, label_after_30_velocity_np, label_after_40_velocity_np,
                label_after_10_angular_velocity_np, label_after_20_angular_velocity_np, label_after_30_angular_velocity_np, label_after_40_angular_velocity_np):
    return Dataset(map_input_tensor, record_input_tensor,
                   label_after_10_velocity_np, label_after_20_velocity_np, label_after_30_velocity_np, label_after_40_velocity_np,
                   label_after_10_angular_velocity_np, label_after_20_angular_velocity_np, label_after_30_angular_velocity_np, label_after_40_angular_velocity_np)

# ========================================================================================== #

# For GMM
def get_embeddings(net, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device,):
    num_samples = len(loader.dataset)
    embeddings_vel_1 = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    embeddings_vel_2 = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    embeddings_vel_3 = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    embeddings_vel_4 = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels_vel_1 = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    labels_vel_2 = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    labels_vel_3 = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    labels_vel_4 = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    embeddings_ang_1 = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    embeddings_ang_2 = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    embeddings_ang_3 = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    embeddings_ang_4 = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels_ang_1 = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    labels_ang_2 = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    labels_ang_3 = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    labels_ang_4 = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for (input_1, input_2), (label_vel_1, label_vel_2, label_vel_3, label_vel_4, label_ang_1, label_ang_2, label_ang_3, label_ang_4) in loader:
            input_1, input_2 = input_1.to(device), input_2.to(device) # map_input_tensor_test, record_input_tensor_test
            label_vel_1, label_vel_2, label_vel_3, label_vel_4, label_ang_1, label_ang_2, label_ang_3, label_ang_4 \
            = label_vel_1.to(device), label_vel_2.to(device), label_vel_3.to(device), label_vel_4.to(device), label_ang_1.to(device), label_ang_2.to(device), label_ang_3.to(device), label_ang_4.to(device) # label: (batch_size)

            if isinstance(net, nn.DataParallel):
                out_vel_1, out_vel_2, out_vel_3, out_vel_4, out_ang_1, out_ang_2, out_ang_3, out_ang_4 = net.module(input_1, input_2)
                out_vel_1, out_vel_2, out_vel_3, out_vel_4, out_ang_1, out_ang_2, out_ang_3, out_ang_4 \
                = net.module.feature_v_1, net.module.feature_v_2, net.module.feature_v_3, net.module.feature_v_4, net.module.feature_y_1, net.module.feature_y_2, net.module.feature_y_3, net.module.feature_y_4
            else:
                out_vel_1, out_vel_2, out_vel_3, out_vel_4, out_ang_1, out_ang_2, out_ang_3, out_ang_4 = net(input_1, input_2) # out: (batch_size, 25)
                out_vel_1, out_vel_2, out_vel_3, out_vel_4, out_ang_1, out_ang_2, out_ang_3, out_ang_4 \
                = net.feature_v_1, net.feature_v_2, net.feature_v_3, net.feature_v_4, net.feature_y_1, net.feature_y_2, net.feature_y_3, net.feature_y_4 # out: (batch_size, 128) NEED TO CHECK

            end = start + len(input_1) # (batch_size)
            embeddings_vel_1[start:end].copy_(out_vel_1, non_blocking=True)
            embeddings_vel_2[start:end].copy_(out_vel_2, non_blocking=True)
            embeddings_vel_3[start:end].copy_(out_vel_3, non_blocking=True)
            embeddings_vel_4[start:end].copy_(out_vel_4, non_blocking=True)
            labels_vel_1[start:end].copy_(label_vel_1, non_blocking=True)
            labels_vel_2[start:end].copy_(label_vel_2, non_blocking=True)
            labels_vel_3[start:end].copy_(label_vel_3, non_blocking=True)
            labels_vel_4[start:end].copy_(label_vel_4, non_blocking=True)
            embeddings_ang_1[start:end].copy_(out_ang_1, non_blocking=True)
            embeddings_ang_2[start:end].copy_(out_ang_2, non_blocking=True)
            embeddings_ang_3[start:end].copy_(out_ang_3, non_blocking=True)
            embeddings_ang_4[start:end].copy_(out_ang_4, non_blocking=True)
            labels_ang_1[start:end].copy_(label_ang_1, non_blocking=True)
            labels_ang_2[start:end].copy_(label_ang_2, non_blocking=True)
            labels_ang_3[start:end].copy_(label_ang_3, non_blocking=True)
            labels_ang_4[start:end].copy_(label_ang_4, non_blocking=True)
            start = end

    return embeddings_vel_1, embeddings_vel_2, embeddings_vel_3, embeddings_vel_4, labels_vel_1, labels_vel_2, labels_vel_3, labels_vel_4, embeddings_ang_1, embeddings_ang_2, embeddings_ang_3, embeddings_ang_4, labels_ang_1, labels_ang_2, labels_ang_3, labels_ang_4

# ========================================================================================== #

JITTERS = [0, torch.finfo(torch.double).tiny] + \
          [10 ** exp for exp in range(-308, 0, 1)] + \
          [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2] # concat three lists
def _centered_cov_torch(x):
    n = x.shape[0]
    if n <= 1:
        return torch.eye(x.shape[1], device=x.device)
        # print("x.shape[1] :", torch.eye(x.shape[1]))
        # exit()
    res = 1 / (n - 1) * x.t().mm(x)
    return res

def gmm_fit(embeddings, labels, num_classes):
    classwise_mean_features = []
    classwise_cov_features = []

    with torch.no_grad():
        for c in range(num_classes):
            class_samples = embeddings[labels == c]
            
            if class_samples.shape[0] == 0: # No Skip
                dim = embeddings.shape[1]
                mean = torch.zeros(dim, device=embeddings.device)
                cov = torch.eye(dim, device=embeddings.device)
            else:
                mean = torch.mean(class_samples, dim=0)
                cov = _centered_cov_torch(class_samples - mean)

            classwise_mean_features.append(mean)
            classwise_cov_features.append(cov)

        classwise_mean_features = torch.stack(classwise_mean_features)
        classwise_cov_features = torch.stack(classwise_cov_features)

    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(classwise_cov_features.shape[1], device=classwise_cov_features.device,).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),) # loc: (16129, 128), covariance_matrix: (16129, 128, 128)
                break #####
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "The parameter covariance_matrix has invalid values" in str(e):
                    continue
            # break

    return gmm, jitter_eps

# ========================================================================================== #
# ========================================================================================== #
# ========================================================================================== #

if __name__ == "__main__":
    timestamp_step_start = time.time()

    # Parsing the arguments
    args = evaluation_args().parse_args()

    # ============================== #
    
    # Setting the seed
    print("[Info] args :", args)
    print("[Info] seed :", args.seed)
    torch.manual_seed(args.seed)

    # Setting the device
    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if cuda else "cpu")
    print("[Info] CUDA :", str(cuda))

    # ============================== #
    
    # Choosing the model to evaluate
    net = models[args.model]()
    
    # ============================== #

    # Using the gpu
    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        
    # ============================== #    

    # Loading the saved model to evaluate
    load_name = "model_dsfp_town03_aug_150200250_300_(yawdiff_no_filter_before_10)"
    print("[Info] load_name :", load_name)
    net.load_state_dict(torch.load("./save/" + load_name + ".pth"))
    net.eval()

    # ============================== #    

    # Drawing the map image for training
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
                draw_wedge_patterned_line(world_map, start_point, end_point, color)
                # cv2.line(world_map, ((lane[i] + compensator) * scale).astype(np.int32), ((lane[i+1] + compensator) * scale).astype(np.int32), color, 4)

    # ============================== #    

    dataset_name = "log_speed_0percent_town03_aug_150200250"
    dataset_directory = "data" + "/" + dataset_name + "/"
    print("[Info] dataset_directory :", dataset_directory)
    number_of_dataset = 501

    # ============================== #

    embeddings_vel_1_list = []
    embeddings_vel_2_list = []
    embeddings_vel_3_list = []
    embeddings_vel_4_list = []
    labels_vel_1_list = []
    labels_vel_2_list = []
    labels_vel_3_list = []
    labels_vel_4_list = []
    embeddings_ang_1_list = []
    embeddings_ang_2_list = []
    embeddings_ang_3_list = []
    embeddings_ang_4_list = []
    labels_ang_1_list = []
    labels_ang_2_list = []
    labels_ang_3_list = []
    labels_ang_4_list = []
    
    keys = ['after_10', 'after_20', 'after_30', 'after_40']
    class_counts_total_velocity = {key: defaultdict(int) for key in keys}
    class_counts_total_yaw_diff = {key: defaultdict(int) for key in keys}
    used_data_counts_total = 0

    for epoch in range(args.runs):
        print("========== Epoch", epoch, "==========")
        timestamp_step_start = time.time()

        # Loading the matrix dataset for preprocessing
        record = np.load(dataset_directory + str(random.randrange(number_of_dataset)) + ".npy") # (5000, number of vehicles spawned, [location.x, locataion.y, rotation.yaw, v.x, v.y]))
        record_index_shuffled = list(range(10, np.shape(record)[0] - 40)) # Discarding the first index and last 60 indices
        random.shuffle(record_index_shuffled)
        
        # ============================== #

        # Sampling the 100 indices from 0 to 4970
        num_index_samples = 70
        num_vehicle_samples = 125 # Vehicles are spawned in random points for each iteration.
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
            
            for cr in combined_record_sampled:
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
                width, height = 2*scale, 5*scale
                points = np.array([[-width/2, -height/2],
                                   [width/2, -height/2],
                                   [width/2, height/2],
                                   [-width/2, height/2]])
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
                cv2.line(world_map_input_copied, before_10_location, before_5_location, (0, 0, 255), 3)
                cv2.line(world_map_input_copied, before_5_location, current_location, (0, 0, 255), 3)

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

            # ============================== #

            # Converting the arrays to tensors for inputs of model
            map_input_tensor = (torch.tensor(np.array(map_input_list), dtype=torch.float32, requires_grad=True).permute(0, 3, 1, 2)).to(device) # (num_vehicle_samples - len(counter_exclude_list), map_cropping_size height, map_cropping_size width, 3 channels) → (num_vehicle_samples - len(counter_exclude_list), 3 channels, map_cropping_size height, map_cropping_size width)
            record_input_tensor = torch.tensor(history_record_sampled_filtered, dtype=torch.float32, requires_grad=True).to(device) # (num_vehicle_samples - len(counter_exclude_list), [location.x, locataion.y, rotation.yaw, v.x, v.y])
            label_after_10_velocity_tensor = torch.tensor(np.array(label_after_10_velocity_list), requires_grad=False).to(device)
            label_after_20_velocity_tensor = torch.tensor(np.array(label_after_20_velocity_list), requires_grad=False).to(device)
            label_after_30_velocity_tensor = torch.tensor(np.array(label_after_30_velocity_list), requires_grad=False).to(device)
            label_after_40_velocity_tensor = torch.tensor(np.array(label_after_40_velocity_list), requires_grad=False).to(device)
            label_after_10_yaw_diff_tensor = torch.tensor(np.array(label_after_10_yaw_diff_list), requires_grad=False).to(device)
            label_after_20_yaw_diff_tensor = torch.tensor(np.array(label_after_20_yaw_diff_list), requires_grad=False).to(device) 
            label_after_30_yaw_diff_tensor = torch.tensor(np.array(label_after_30_yaw_diff_list), requires_grad=False).to(device)
            label_after_40_yaw_diff_tensor = torch.tensor(np.array(label_after_40_yaw_diff_list), requires_grad=False).to(device)
            
            # ============================== #

            dataset = get_dataset(map_input_tensor, record_input_tensor,
                                  label_after_10_velocity_tensor, label_after_20_velocity_tensor, label_after_30_velocity_tensor, label_after_40_velocity_tensor,
                                  label_after_10_yaw_diff_tensor, label_after_20_yaw_diff_tensor, label_after_30_yaw_diff_tensor, label_after_40_yaw_diff_tensor)
            dataloader = DataLoader(dataset, batch_size=4096, shuffle=False)

            # ============================== #

            embeddings_vel_1, embeddings_vel_2, embeddings_vel_3, embeddings_vel_4, \
            labels_vel_1, labels_vel_2, labels_vel_3, labels_vel_4, \
            embeddings_ang_1, embeddings_ang_2, embeddings_ang_3, embeddings_ang_4, \
            labels_ang_1, labels_ang_2, labels_ang_3, labels_ang_4 = get_embeddings(net, dataloader, num_dim=model_to_num_dim[args.model], dtype=torch.double, device=device, storage_device=device,)

            # ============================== #

            used_data_counts_total += labels_vel_1.size(0)

            labels_vel_list = [labels_vel_1, labels_vel_2, labels_vel_3, labels_vel_4]
            labels_ang_list = [labels_ang_1, labels_ang_2, labels_ang_3, labels_ang_4]

            for i, key in enumerate(keys):
                class_counts = torch.bincount(labels_vel_list[i], minlength=num_velocity_classes)
                for class_idx, count in enumerate(class_counts):
                    class_counts_total_velocity[key][f'{key}_class_{class_idx}'] += count.item()
                    
            for i, key in enumerate(keys):
                class_counts = torch.bincount(labels_ang_list[i], minlength=num_yaw_diff_classes)
                for class_idx, count in enumerate(class_counts):
                    class_counts_total_yaw_diff[key][f'{key}_class_{class_idx}'] += count.item() 

            # ============================== #

            embeddings_vel_1_list.append(embeddings_vel_1)
            embeddings_vel_2_list.append(embeddings_vel_2)
            embeddings_vel_3_list.append(embeddings_vel_3)
            embeddings_vel_4_list.append(embeddings_vel_4)
            labels_vel_1_list.append(labels_vel_1)
            labels_vel_2_list.append(labels_vel_2)
            labels_vel_3_list.append(labels_vel_3)
            labels_vel_4_list.append(labels_vel_4)
            embeddings_ang_1_list.append(embeddings_ang_1)
            embeddings_ang_2_list.append(embeddings_ang_2)
            embeddings_ang_3_list.append(embeddings_ang_3)
            embeddings_ang_4_list.append(embeddings_ang_4)
            labels_ang_1_list.append(labels_ang_1)
            labels_ang_2_list.append(labels_ang_2)
            labels_ang_3_list.append(labels_ang_3)
            labels_ang_4_list.append(labels_ang_4)

        timestamp_step_end = time.time()
        print(f"[Time] {timestamp_step_end - timestamp_step_start:.1f} seconds\n")

    embeddings_vel_1_tensor = torch.cat(embeddings_vel_1_list, dim=0)
    embeddings_vel_2_tensor = torch.cat(embeddings_vel_2_list, dim=0)
    embeddings_vel_3_tensor = torch.cat(embeddings_vel_3_list, dim=0)
    embeddings_vel_4_tensor = torch.cat(embeddings_vel_4_list, dim=0)
    labels_vel_1_tensor = torch.cat(labels_vel_1_list, dim=0)
    labels_vel_2_tensor = torch.cat(labels_vel_2_list, dim=0)
    labels_vel_3_tensor = torch.cat(labels_vel_3_list, dim=0)
    labels_vel_4_tensor = torch.cat(labels_vel_4_list, dim=0)
    embeddings_ang_1_tensor = torch.cat(embeddings_ang_1_list, dim=0)
    embeddings_ang_2_tensor = torch.cat(embeddings_ang_2_list, dim=0)
    embeddings_ang_3_tensor = torch.cat(embeddings_ang_3_list, dim=0)
    embeddings_ang_4_tensor = torch.cat(embeddings_ang_4_list, dim=0)
    labels_ang_1_tensor = torch.cat(labels_ang_1_list, dim=0)
    labels_ang_2_tensor = torch.cat(labels_ang_2_list, dim=0)
    labels_ang_3_tensor = torch.cat(labels_ang_3_list, dim=0)
    labels_ang_4_tensor = torch.cat(labels_ang_4_list, dim=0)

    # ============================== #

    gaussian_models_velocity_1, _ = gmm_fit(embeddings=embeddings_vel_1_tensor, labels=labels_vel_1_tensor, num_classes=num_velocity_classes)
    gaussian_models_velocity_2, _ = gmm_fit(embeddings=embeddings_vel_2_tensor, labels=labels_vel_2_tensor, num_classes=num_velocity_classes)
    gaussian_models_velocity_3, _ = gmm_fit(embeddings=embeddings_vel_3_tensor, labels=labels_vel_3_tensor, num_classes=num_velocity_classes)
    gaussian_models_velocity_4, _ = gmm_fit(embeddings=embeddings_vel_4_tensor, labels=labels_vel_4_tensor, num_classes=num_velocity_classes)
    gaussian_models_angular_velocity_1, _ = gmm_fit(embeddings=embeddings_ang_1_tensor, labels=labels_ang_1_tensor, num_classes=num_yaw_diff_classes)
    gaussian_models_angular_velocity_2, _ = gmm_fit(embeddings=embeddings_ang_2_tensor, labels=labels_ang_2_tensor, num_classes=num_yaw_diff_classes)
    gaussian_models_angular_velocity_3, _ = gmm_fit(embeddings=embeddings_ang_3_tensor, labels=labels_ang_3_tensor, num_classes=num_yaw_diff_classes)
    gaussian_models_angular_velocity_4, _ = gmm_fit(embeddings=embeddings_ang_4_tensor, labels=labels_ang_4_tensor, num_classes=num_yaw_diff_classes)
    
    velocity_parameters = {'velocity_1': {'mean': gaussian_models_velocity_1.mean, 'covariance': gaussian_models_velocity_1.covariance_matrix},
                           'velocity_2': {'mean': gaussian_models_velocity_2.mean, 'covariance': gaussian_models_velocity_2.covariance_matrix},
                           'velocity_3': {'mean': gaussian_models_velocity_3.mean, 'covariance': gaussian_models_velocity_3.covariance_matrix},
                           'velocity_4': {'mean': gaussian_models_velocity_4.mean, 'covariance': gaussian_models_velocity_4.covariance_matrix}}
    angular_velocity_parameters = {'yaw_diff_1': {'mean': gaussian_models_angular_velocity_1.mean, 'covariance': gaussian_models_angular_velocity_1.covariance_matrix},
                                   'yaw_diff_2': {'mean': gaussian_models_angular_velocity_2.mean, 'covariance': gaussian_models_angular_velocity_2.covariance_matrix},
                                   'yaw_diff_3': {'mean': gaussian_models_angular_velocity_3.mean, 'covariance': gaussian_models_angular_velocity_3.covariance_matrix},
                                   'yaw_diff_4': {'mean': gaussian_models_angular_velocity_4.mean, 'covariance': gaussian_models_angular_velocity_4.covariance_matrix}}

    torch.save(velocity_parameters, 'gmm_velocity_parameters.pth')
    torch.save(angular_velocity_parameters, 'gmm_yaw_diff_parameters.pth')

    # ============================== #

    with open('class_counts_total_velocity.pkl', 'wb') as f:
        pickle.dump(class_counts_total_velocity, f)
    with open('class_counts_total_yaw_diff.pkl', 'wb') as f:
        pickle.dump(class_counts_total_yaw_diff, f)

    with open('class_counts_total_velocity.pkl', 'rb') as f:
        class_counts_total_velocity_loaded = pickle.load(f)
    with open('class_counts_total_yaw_diff.pkl', 'rb') as f:
        class_counts_total_yaw_diff_loaded = pickle.load(f)

    class_total_velocity_sum = sum(class_counts_total_velocity_loaded['after_10'].values())
    class_total_yaw_diff_sum = sum(class_counts_total_yaw_diff_loaded['after_10'].values())

    print(f"[Info] used_data_counts_total : {used_data_counts_total}")
    print(f"[Info] class_total_velocity_sum : {class_total_velocity_sum}")
    print(f"[Info] class_total_yaw_diff_sum : {class_total_yaw_diff_sum}")
    print("")
    print("[Info] class_counts_total_velocity :")
    for key in keys:
        class_counts_total_velocity_list = [f"({j}){class_counts_total_velocity_loaded[key][f'{key}_class_{j}']}" for j in range(num_velocity_classes)]
        print(f"{key}_class : {class_counts_total_velocity_list}")
    print("")
    print("[Info] class_counts_total_yaw_diff :")
    for key in keys:
        class_counts_total_yaw_diff_list = [f"({j}){class_counts_total_yaw_diff_loaded[key][f'{key}_class_{j}']}" for j in range(num_yaw_diff_classes)]
        print(f"{key}_class : {class_counts_total_yaw_diff_list}")

    timestamp_step_end = time.time()
    print("[Info] GMM Fitting Complete")
    print(f"[Time] {timestamp_step_end - timestamp_step_start:.1f} seconds\n")