import os
import csv
import json
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def getInputData(data_path):
    u_val = []
    new_u_val = []
    y_val = []
    x_val = []

    # with open('C://Users//selen//OneDrive//Desktop//master//SSM_carPredictor//left_wheel_data.csv', mode='r', encoding='utf-8') as file:
    #     reader = csv.reader(file)
    #     next(reader)  # Skip header
    #     for row in reader:
    #         u_val.append(float(row[1]))

    # with open('C://Users//selen//OneDrive//Desktop//master//SSM_carPredictor//newCarData.csv', mode='r', encoding='utf-8') as file:
    #     reader = csv.reader(file)
    #     next(reader)  # Skip header
    #     for row in reader:
    #         time = int(float(row[0]))
    #         y_val.append(float(row[1]))
    #         new_u_val.append(float(u_val[time]))

    # with open('C://Users//selen//OneDrive//Desktop//master//SSM_carPredictor//x_data.csv', mode='r', encoding='utf-8') as file:
    #     reader = csv.reader(file)
    #     next(reader)  # Skip header
    #     index = 0
    #     count = 0
    #     for row in reader:
    #         if count > len(new_u_val):
    #             break
    #         if index % 100 == 0: # 0.001 -> 0.1s
    #             count += 1
    #             # Each x_data row has 4 values
    #             x_val.append([[float(row[0])], [float(row[1])], [float(row[2])], [float(row[3])]])
    #         index += 1

    with open(data_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            u_val.append([[float(row[1])], [float(row[2])], [float(row[3])], [float(row[4])]])
            angle = np.radians(float(row[8]))
            x_val.append([[float(row[5])], [float(row[7])], [np.sin(angle)], [np.cos(angle)]])

    u_val, u_scaler = normalize(np.array(u_val), "u")
    x_val, x_scaler = normalize(np.array(x_val), "x")

    u_val = u_val[:-1]
    x_next_val = x_val[1:]
    x_val = x_val[:-1]

    train_size = int(0.8 * len(u_val))
    train_u = u_val[:train_size]
    # train_y = y_val[:train_size]
    train_x = x_val[:train_size]
    train_x_next = x_next_val[:train_size]

    test_u = u_val[train_size:]
    # test_y = y_val[train_size:]
    test_x = x_val[train_size:]
    test_x_next = x_next_val[train_size:]

    return train_u, train_x, train_x_next, test_u, test_x, test_x_next, u_scaler, x_scaler

def loadConfig():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def computeTarget(u_tensor, x_data, A, B, dt=0.01):
    x_dot = torch.bmm(A, x_data) + torch.bmm(B, u_tensor)  # x' = Ax + Bu
    # y_pred = torch.bmm(C, x_data) + torch.bmm(D, u_tensor.unsqueeze(2))  # y = Cx + Du
    next_x = x_data + x_dot * dt  # Euler integration: x_next = x + x' * dt
    # y_pred = y_pred.squeeze(2)  # reshape y_pred to match y_tensor

    return next_x

def normalize(data, name="data"):
    if name == "x":
        # normalize on first 2 features only (x,z positions)
        scaler = StandardScaler()
        data[:, :, :2] = scaler.fit_transform(data[:, :, :2].reshape(-1, 2)).reshape(data[:, :, :2].shape)
    else:
        scaler = StandardScaler()
        original_shape = data.shape
        data_reshaped = data.reshape(-1, original_shape[-1])
        data_normalized = scaler.fit_transform(data_reshaped)
        data = data_normalized.reshape(original_shape)
    return data, scaler

def angleToDegree(data):
    sin_component = data[:, 2, :]
    cos_component = data[:, 3, :]
    angles = np.arctan2(sin_component, cos_component)
    angles_degrees = np.degrees(angles)
    angles_degrees = (angles_degrees + 360) % 360  # Normalize to [0, 360)
    data[:, 2, :] = angles_degrees
    data[:, 3, :] = 0  # Optionally set the cosine component to zero or remove it
    return data