# state space model based neural network model for car price prediction
import os
import torch
from carPredictor import CarPredictor
from torch.utils.tensorboard import SummaryWriter
from utils import getInputData, loadConfig, computeTarget
import matplotlib.pyplot as plt

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    mode = input("Enter mode (1 or 2): ")
    return mode

def trainModel(model, u_tensor, y_tensor, x_tensor, epochs=100, learning_rate=0.001, name="model"):
    if os.path.exists(f'../{config["name"]}.pth'):
        print(f"Model {config['name']} already exists. Loading existing model.")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
    else:
        print(f"Using new model {config['name']}.")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_temp = x_tensor.view(x_tensor.size(0), -1) # flatten x_tensor
    input_data = torch.cat((u_tensor, x_temp), dim=1).to(device)
    x_tensor = x_tensor.to(device)
    u_tensor = u_tensor.to(device)
    y_tensor = y_tensor.to(device)
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(input_data)
        y_pred, next_x = computeTarget(u_tensor, outputs, x_tensor)
        loss_1 = criterion(y_pred, y_tensor)
        # remove first data point from x_tensor
        x_tensor_temp = x_tensor[1:, :, :]
        # remove last data point from next_x
        next_x_temp = next_x[:-1, :, :]
        loss_2 = criterion(next_x_temp, x_tensor_temp)
        loss = loss_1 + loss_2
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch+1)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models', name), exist_ok=True)
        if (epoch + 1) % 1000 == 0:
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', 'models', name, f'{epoch+1}.pth'))

    writer.close()
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', f'{name}.pth'))
    print(f'Model saved as {name}.pth')


def runInference(model, u_tensor, y_tensor, x_tensor, name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    x_temp = x_tensor.view(x_tensor.size(0), -1) # flatten x_tensor
    input_data = torch.cat((u_tensor, x_temp), dim=1).to(device)
    x_tensor = x_tensor.to(device)
    x_data = x_tensor[0:1, :, :]  # start with the first
    u_tensor = u_tensor.to(device)
    y_tensor = y_tensor.to(device)


    with torch.no_grad():
        output = model(input_data)  # Send one data point at a time
        y_pred, next_x = computeTarget(u_tensor, output, x_tensor)

    difference = y_pred - y_tensor
    # get the mean of each output column
    mean_output = torch.mean(output, dim=0)
    print(mean_output)

    plt.figure(figsize=(10,5))
    plt.plot(difference.cpu().numpy(), label='Difference (y - y_pred)', color='red')
    plt.legend()
    plt.title('Inference Difference')
    plt.xlabel('Time Step')
    plt.ylabel('Difference')
    plt.savefig(f'../{name}.png')


if __name__ == "__main__":
    mode = selectMode()
    u_val, y_val, x_val = getInputData()
    config = loadConfig()
    u_tensor = torch.tensor(u_val).float().unsqueeze(1)
    y_tensor = torch.tensor(y_val).float().unsqueeze(1)
    x_tensor = torch.tensor(x_val).float()
    model = CarPredictor(
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout']
    )

    if mode == "1":
        print("Training model.")
        trainModel(model, u_tensor, y_tensor, x_tensor, config['model']['epochs'], config['model']['learning_rate'], config['name'])

    elif mode == "2":
        print("Inference")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
        runInference(model, u_tensor, y_tensor, x_tensor, config['name'])