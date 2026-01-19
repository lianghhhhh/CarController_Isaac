import os
import csv
import json
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(data):
    current_pos = data[2:5]  # pos_x, pos_y, angle
    target_pos = data[5:7]   # target_x, target_y
    
    delta_x = target_pos[0] - current_pos[0]
    delta_y = target_pos[1] - current_pos[1]
    d_angle = data[7] # already calculate as error angle in radians
    d_angle = (d_angle + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

    local_dx = (delta_x * np.cos(-current_pos[2]) - delta_y * np.sin(-current_pos[2]))
    local_dy = (delta_x * np.sin(-current_pos[2]) + delta_y * np.cos(-current_pos[2]))

    data = data[0:2] + [local_dx, local_dy, d_angle]

    # scaler = joblib.load('/home/liangh/car_data_collect/input_scaler.save')
    # data = np.array(data).reshape(1, -1)
    # data = scaler.transform(data)
    return np.array(data).reshape(1, -1)

def calActualState(pos_x, pos_y, angle, delta_state):
    delta_x = delta_state[0]
    delta_y = delta_state[1]
    d_angle = delta_state[2]

    global_dx = delta_x * np.cos(angle) - delta_y * np.sin(angle)
    global_dy = delta_x * np.sin(angle) + delta_y * np.cos(angle)

    new_x = pos_x + global_dx
    new_y = pos_y + global_dy
    new_angle = angle + d_angle
    new_angle = (new_angle + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

    return [new_x, new_y, new_angle]

def denormalize(data):
    scaler = joblib.load('/home/liangh/car_data_collect/output_scaler.save')
    data = np.array(data).reshape(1, -1)
    data = scaler.inverse_transform(data)
    return data.flatten().tolist()

def getInputData(data_path):
    target_vels = []
    car_pos = []
    current_vels = []
    target_pos = []

    with open(data_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            target_vels.append([float(row[0]), float(row[1])])  # target velocity
            current_vels.append([float(row[2]), float(row[3])])  # current velocity
            car_angle = float(row[6])
            if car_angle > 360:
                car_angle -= 360
            car_pos.append([float(row[4]), float(row[5]), car_angle])  # position x, z, angle
            target_pos.append([float(row[7]), float(row[8]), float(row[9])])  # target position x, z, angle
    target_vels = target_vels[1:]  # remove first entry to align with state deltas
    current_vels = current_vels[:-1]  # remove last entry to align with state deltas
    pos_delta = []
    for i in range(len(car_pos) - 1):
        global_dx = target_pos[i][0] - car_pos[i][0]
        global_dz = target_pos[i][1] - car_pos[i][1]
        d_angle = target_pos[i][2] - car_pos[i][2]
        if d_angle > 180:
            d_angle -= 360
        elif d_angle < -180:
            d_angle += 360
        
        # convert global deltas to local car frame
        angle_rad = np.radians(car_pos[i][2])
        local_dx = global_dx * np.cos(-angle_rad) - global_dz * np.sin(-angle_rad)
        local_dz = global_dx * np.sin(-angle_rad) + global_dz * np.cos(-angle_rad)
        pos_delta.append([local_dx, local_dz, d_angle])

    input_data = np.hstack((current_vels, pos_delta))  # concatenate current velocity and position deltas
    output_data = np.array(target_vels)

    input_data, input_scaler = normalize(input_data)
    output_data, output_scaler = normalize(output_data)

    joblib.dump(input_scaler, os.path.join(os.path.dirname(__file__), '..', 'input_scaler.save'))
    joblib.dump(output_scaler, os.path.join(os.path.dirname(__file__), '..', 'output_scaler.save'))

    train_size = int(0.8 * len(target_vels))
    train_input = input_data[:train_size]
    train_output = output_data[:train_size]

    test_input = input_data[train_size:]
    test_output = output_data[train_size:]
    return train_input, train_output, test_input, test_output, input_scaler, output_scaler

def loadConfig():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def normalize(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler

def splitTrainVal(input_data, output_data, val_ratio=0.1):
    total_size = len(input_data)
    val_size = int(total_size * val_ratio)
    train_input = input_data[:-val_size]
    train_output = output_data[:-val_size]
    val_input = input_data[-val_size:]
    val_output = output_data[-val_size:]
    return train_input, train_output, val_input, val_output