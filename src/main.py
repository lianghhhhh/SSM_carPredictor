# state space model based neural network model for car state prediction
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from carPredictor import CarPredictor
from torch.utils.tensorboard import SummaryWriter
from utils import getInputData, loadConfig, computeTarget, angleToDegree

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    mode = input("Enter mode (1 or 2): ")
    return mode

def trainModel(model, u_tensor, x_tensor, x_next_tensor, epochs=100, learning_rate=0.001, name="model"):
    if os.path.exists(f'../{config["name"]}.pth'):
        print(f"Model {config['name']} already exists. Loading existing model.")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
    else:
        print(f"Using new model {config['name']}.")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_tensor = x_tensor.to(device)
    u_tensor = u_tensor.to(device)
    x_next_tensor = x_next_tensor.to(device)
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        for i in range(0, u_tensor.size(0), 32):  # mini-batch training
            batch_u = u_tensor[i:i+32]
            batch_x = x_tensor[i:i+32]
            batch_x_next = x_next_tensor[i:i+32]

            model.train()
            optimizer.zero_grad()
            A, B = model(batch_u, batch_x)
            next_x_pred = computeTarget(batch_u, batch_x, A, B)
            # loss_1 = criterion(y_pred, batch_y)
            loss_2 = criterion(next_x_pred, batch_x_next)
            loss = loss_2
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch+1)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models', name), exist_ok=True)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', 'models', name, f'{epoch+1}.pth'))

    writer.close()
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', f'{name}.pth'))
    print(f'Model saved as {name}.pth')


def runInference(model, u_tensor, x_tensor, x_next_tensor, x_scaler, name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    x_tensor = x_tensor.to(device)
    u_tensor = u_tensor.to(device)
    x_pred_list = []

    with torch.no_grad():
        for i in range(u_tensor.size(0)):  # time steps
            A, B = model(u_tensor[i:i+1], x_tensor[i:i+1])
            next_x_pred = computeTarget(u_tensor[i:i+1], x_tensor[i:i+1], A, B)
            x_pred_list.append(next_x_pred)
            # x_t = next_x_pred  # update state for next time step

    x_pred = torch.cat(x_pred_list, dim=0)
    x_pred = x_pred.cpu().numpy()
    x_pred[:, :, :2] = x_scaler.inverse_transform(x_pred[:, :, :2].reshape(-1, 2)).reshape(x_pred[:, :, :2].shape)

    x_next_tensor = x_next_tensor.cpu().numpy()
    x_next_tensor[:, :, :2] = x_scaler.inverse_transform(x_next_tensor[:, :, :2].reshape(-1, 2)).reshape(x_next_tensor[:, :, :2].shape)
    
    x_pred = angleToDegree(x_pred)
    x_next_tensor = angleToDegree(x_next_tensor)

    difference = x_pred - x_next_tensor
    print(x_pred)
    print(x_next_tensor)

    difference = x_pred - x_next_tensor
    print(difference)

    # plot all in one figure
    time_steps = np.arange(x_pred.shape[0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    for i in range(3):
        difference = x_pred[:, i]-x_next_tensor[:, i]
        if(i==2):
            # wrap around for angle difference
            difference = (difference + 180) % 360 - 180
        axs[i].plot(time_steps, difference, label='Difference', color='red')
        axs[i].set_title(f'Dimension {i+1}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(f'../{name}.png')


if __name__ == "__main__":
    mode = selectMode()
    config = loadConfig()
    train_u, train_x, train_x_next, test_u, test_x, test_x_next, u_scaler, x_scaler = getInputData(config['data'])

    train_u = torch.tensor(train_u).float()
    train_x = torch.tensor(train_x).float()
    train_x_next = torch.tensor(train_x_next).float()
    test_u = torch.tensor(test_u).float()
    test_x = torch.tensor(test_x).float()
    test_x_next = torch.tensor(test_x_next).float()

    print("Sample")
    print("u:", train_u[0:5], "shape:", train_u.shape)
    print("x:", train_x[0:5], "shape:", train_x.shape)
    print("x_next:", train_x_next[0:5], "shape:", train_x_next.shape)

    model = CarPredictor(
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout']
    )

    if mode == "1":
        print("Training model.")
        trainModel(model, train_u, train_x, train_x_next, config['model']['epochs'], config['model']['learning_rate'], config['name'])

    elif mode == "2":
        print("Inference")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
        runInference(model, test_u, test_x, test_x_next, x_scaler, config['name'])