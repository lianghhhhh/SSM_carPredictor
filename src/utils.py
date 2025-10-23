import os
import csv
import json
import torch

def getInputData():
    u_val = []
    new_u_val = []
    y_val = []
    x_val = []

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

    with open('C://Users//selen//OneDrive//Desktop//master//SSM_carPredictor//x_data.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        index = 0
        count = 0
        for row in reader:
            if count > len(new_u_val):
                break
            if index % 100 == 0: # 0.001 -> 0.1s
                count += 1
                # Each x_data row has 4 values
                x_val.append([[float(row[0])], [float(row[1])], [float(row[2])], [float(row[3])]])
            index += 1
    
    x_next_val = x_val[1:]
    x_val = x_val[:-1]

    train_size = int(0.8 * len(new_u_val))
    train_u = new_u_val[:train_size]
    train_y = y_val[:train_size]
    train_x = x_val[:train_size]
    train_x_next = x_next_val[:train_size]

    test_u = new_u_val[train_size:]
    test_y = y_val[train_size:]
    test_x = x_val[train_size:]

    return train_u, train_y, train_x, train_x_next, test_u, test_y, test_x

def loadConfig():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def computeTarget(u_tensor, x_data, A, B, C, D, dt=0.1):
    x_dot = torch.bmm(A, x_data) + torch.bmm(B, u_tensor.unsqueeze(2))  # x' = Ax + Bu
    y_pred = torch.bmm(C, x_data) + torch.bmm(D, u_tensor.unsqueeze(2))  # y = Cx + Du
    next_x = x_data + x_dot * dt  # Euler integration: x_next = x + x' * dt
    y_pred = y_pred.squeeze(2)  # reshape y_pred to match y_tensor

    return y_pred, next_x
