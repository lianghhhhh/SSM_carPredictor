# state space model based neural network model for car price prediction
import os
import torch
from carPredictor import CarPredictor
from torch.utils.tensorboard import SummaryWriter
from utils import getInputData, loadConfig, computeTarget

def selectMode():
    print("Select mode:")
    print("1. Train")
    print("2. Inference")
    mode = input("Enter mode (1 or 2): ")
    return mode

def trainModel(model, u_tensor, y_tensor, epochs=100, learning_rate=0.001, name="model"):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = torch.cat((u_tensor, y_tensor), dim=1).to(device)
    x_data = torch.zeros(u_tensor.size(0), 4).to(device)  # initial state x:[0,0,0,0]
    u_tensor = u_tensor.to(device)
    y_tensor = y_tensor.to(device)
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(input_data)
        y_pred, next_x = computeTarget(u_tensor, outputs, x_data)
        x_data = next_x.detach()  # detach to prevent gradient accumulation
        loss = criterion(y_pred, y_tensor)
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

    if os.path.exists(f'../{config["name"]}.pth'):
        print(f"Model {config['name']} already exists. Loading existing model.")
        model.load_state_dict(torch.load(f'../{config["name"]}.pth'))
    else:
        print(f"Using new model {config['name']}.")

    if mode == "1":
        print("Training model.")
        trainModel(model, u_tensor, y_tensor, config['model']['epochs'], config['model']['learning_rate'], config['name'])