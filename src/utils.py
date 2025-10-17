import os
import csv
import json
import torch

def getInputData():
    u_val = []
    new_u_val = []
    y_val = []

    with open('C://Users//selen//OneDrive//Desktop//master//SSM_carPredictor//left_wheel_data.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            u_val.append(float(row[1]))

    with open('C://Users//selen//OneDrive//Desktop//master//SSM_carPredictor//newCarData.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            time = int(float(row[0]))
            y_val.append(float(row[1]))
            new_u_val.append(float(u_val[time]))

    return new_u_val, y_val

def loadConfig():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def computeTarget(u_tensor, matrix_outputs, x_data):
    # state space model: x' = Ax + Bu, y = Cx + Du, matrix_outputs is the 25-dim output from the model
    A = matrix_outputs[:, 0:16].reshape(-1, 4, 4)  # 4x4 matrix A
    B = matrix_outputs[:, 16:20].reshape(-1, 4, 1)  # 4x1 matrix B
    C = matrix_outputs[:, 20:24].reshape(-1, 1, 4)  # 1x4 matrix C
    D = matrix_outputs[:, 24:25].reshape(-1, 1, 1)  # 1x1 matrix D

    delta_x = torch.bmm(A, x_data.unsqueeze(2)) + torch.bmm(B, u_tensor.unsqueeze(2))  # x' = Ax + Bu
    y_pred = torch.bmm(C, delta_x) + torch.bmm(D, u_tensor.unsqueeze(2))  # y = Cx + Du

    next_x = delta_x.squeeze(2) + x_data  # next state x = x + delta_x
    y_pred = y_pred.squeeze(2)  # reshape y_pred to match y_tensor

    return y_pred, next_x
