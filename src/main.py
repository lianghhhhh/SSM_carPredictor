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

def trainModel(model, u_tensor, y_tensor, epochs=100, learning_rate=0.001, name="model"):
    if os.path.exists(f'../{config["name"]}.pth'):
        print(f"Model {config['name']} already exists. Loading existing model.")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
    else:
        print(f"Using new model {config['name']}.")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = torch.cat((u_tensor, y_tensor), dim=1).to(device)
    x_data = torch.zeros(1, 4).to(device) # initial state x:[0,0,0,0]
    u_tensor = u_tensor.to(device)
    y_tensor = y_tensor.to(device)
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for i in range(input_data.size(0)):
            output = model(input_data[i:i+1])  # Send one data point at a time
            y_pred, next_x = computeTarget(u_tensor[i:i+1], output, x_data)
            if i >= input_data.size(0) - 1:
                break
            x_data = y_tensor[i].unsqueeze(0).repeat(1, 4)  # next state x is the actual current output
            loss_1 = criterion(y_pred, y_tensor[i:i+1])
            loss_2 = criterion(next_x, x_data)
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


def runInference(model, u_tensor, y_tensor, name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    input_data = torch.cat((u_tensor, y_tensor), dim=1).to(device)
    x_data = torch.zeros(u_tensor.size(0), 4).to(device)  # initial state x:[0,0,0,0]
    u_tensor = u_tensor.to(device)
    y_tensor = y_tensor.to(device)
    y_preds = []

    with torch.no_grad():
        for i in range(input_data.size(0)):
            output = model(input_data[i:i+1])  # Send one data point at a time
            y_pred, next_x = computeTarget(u_tensor[i:i+1], output, x_data)
            y_preds.append(y_pred.item())
            x_data = next_x
            # Update x_data for the next iteration

    print(output)


    plt.figure(figsize=(10,5))
    plt.plot(y_tensor.cpu().numpy(), label='Actual')
    plt.plot(y_preds, label='Predicted')
    plt.legend()
    plt.title('Inference Results')
    plt.xlabel('Time Step')
    plt.ylabel('Output')
    plt.show()


if __name__ == "__main__":
    mode = selectMode()
    u_val, y_val = getInputData()
    config = loadConfig()
    u_tensor = torch.tensor(u_val).float().unsqueeze(1)
    y_tensor = torch.tensor(y_val).float().unsqueeze(1)
    model = CarPredictor(
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout']
    )

    if mode == "1":
        print("Training model.")
        trainModel(model, u_tensor, y_tensor, config['model']['epochs'], config['model']['learning_rate'], config['name'])

    elif mode == "2":
        print("Inference")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
        runInference(model, u_tensor, y_tensor, config['name'])