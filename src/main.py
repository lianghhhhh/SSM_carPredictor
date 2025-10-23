# state space model based neural network model for car price prediction
import os
import torch
from carPredictor import CarPredictor
from torch.utils.tensorboard import SummaryWriter
from utils import getInputData, loadConfig, computeTarget
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    mode = input("Enter mode (1 or 2): ")
    return mode

def trainModel(model, u_tensor, y_tensor, x_tensor, x_next_tensor, epochs=100, learning_rate=0.001, name="model"):
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
    y_tensor = y_tensor.to(device)
    x_next_tensor = x_next_tensor.to(device)
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        for i in range(0, u_tensor.size(0), 32):  # mini-batch training
            batch_u = u_tensor[i:i+32]
            batch_y = y_tensor[i:i+32]
            batch_x = x_tensor[i:i+32]
            batch_x_next = x_next_tensor[i:i+32]

            model.train()
            optimizer.zero_grad()
            A, B, C, D = model(batch_u, batch_x)
            y_pred, next_x = computeTarget(batch_u, batch_x, A, B, C, D)
            loss_1 = criterion(y_pred, batch_y)
            loss_2 = criterion(next_x, batch_x_next)
            loss = loss_1 + loss_2
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


def runInference(model, u_tensor, y_tensor, x_tensor, y_scaler, name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    x_tensor = x_tensor.to(device)
    u_tensor = u_tensor.to(device)
    y_tensor = y_tensor.to(device)
    x_t = x_tensor[0:1, :, :]  # start with the first time step
    y_pred_list = []

    with torch.no_grad():
        for i in range(u_tensor.size(0)):  # time steps
            A, B, C, D = model(u_tensor[i:i+1], x_t)
            y_pred, next_x = computeTarget(u_tensor[i:i+1], x_t, A, B, C, D)
            y_pred_list.append(y_pred)
            x_t = next_x  # update state for next time step

    y_pred = torch.cat(y_pred_list, dim=0)
    y_pred_np = y_pred.cpu().numpy()
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_np)
    y_tensor_np = y_tensor.cpu().numpy()
    y_tensor_unscaled = y_scaler.inverse_transform(y_tensor_np)

    difference = y_pred_unscaled - y_tensor_unscaled
    print(y_pred_unscaled)
    print(y_tensor_unscaled)
    print(difference)

    plt.figure(figsize=(10,5))
    plt.plot(difference, label='Difference (y - y_pred)', color='red')
    plt.legend()
    plt.title('Inference Difference')
    plt.xlabel('Time Step')
    plt.ylabel('Difference')
    plt.savefig(f'../{name}.png')


if __name__ == "__main__":
    mode = selectMode()
    train_u, train_y, train_x, train_x_next, test_u, test_y, test_x = getInputData()
    config = loadConfig()
    train_u = torch.tensor(train_u).float().unsqueeze(1)
    train_y = torch.tensor(train_y).float().unsqueeze(1)
    train_x = torch.tensor(train_x).float()
    train_x_next = torch.tensor(train_x_next).float()
    test_u = torch.tensor(test_u).float().unsqueeze(1)
    test_y = torch.tensor(test_y).float().unsqueeze(1)
    test_x = torch.tensor(test_x).float()

    # normalization
    # 2. Reshape data for scalers (N, features)
    # x data is (N, 4, 1), reshape to (N, 4)
    train_x_np = np.array(train_x).reshape(-1, 4)
    train_x_next_np = np.array(train_x_next).reshape(-1, 4)
    test_x_np = np.array(test_x).reshape(-1, 4)
    
    # u/y data are (N,), reshape to (N, 1)
    train_u_np = np.array(train_u).reshape(-1, 1)
    train_y_np = np.array(train_y).reshape(-1, 1)
    test_u_np = np.array(test_u).reshape(-1, 1)
    test_y_np = np.array(test_y).reshape(-1, 1)


    # 3. Create and FIT scalers *ONLY ON TRAINING DATA*
    u_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler = StandardScaler()

    u_scaler.fit(train_u_np)
    y_scaler.fit(train_y_np)
    x_scaler.fit(train_x_np) # x and x_next use the same scaler

    # 4. TRANSFORM all data (train and test)
    train_u_scaled = u_scaler.transform(train_u_np)
    train_y_scaled = y_scaler.transform(train_y_np)
    train_x_scaled = x_scaler.transform(train_x_np)
    train_x_next_scaled = x_scaler.transform(train_x_next_np)
    
    test_u_scaled = u_scaler.transform(test_u_np)
    test_y_scaled = y_scaler.transform(test_y_np)
    test_x_scaled = x_scaler.transform(test_x_np)

    # 5. Convert to Tensors
    train_u = torch.tensor(train_u_scaled).float()
    train_y = torch.tensor(train_y_scaled).float()
    test_u = torch.tensor(test_u_scaled).float()
    test_y = torch.tensor(test_y_scaled).float()

    # Reshape x tensors back to (N, 4, 1) for the model
    train_x = torch.tensor(train_x_scaled).float().unsqueeze(-1)
    train_x_next = torch.tensor(train_x_next_scaled).float().unsqueeze(-1)
    test_x = torch.tensor(test_x_scaled).float().unsqueeze(-1)

    model = CarPredictor(
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout']
    )

    if mode == "1":
        print("Training model.")
        trainModel(model, train_u, train_y, train_x, train_x_next, config['model']['epochs'], config['model']['learning_rate'], config['name'])

    elif mode == "2":
        print("Inference")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
        runInference(model, test_u, test_y, test_x, y_scaler, config['name'])