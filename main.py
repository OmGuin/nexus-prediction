from config import DEVICE, EPOCHS, LEARNING_RATE
"""
main.py
This script trains and tests an IRPredictor model for regression tasks. It utilizes PyTorch for 
model definition, training, and testing, and exports the trained model in both `.pth` and `.onnx` formats.
Modules:
- config: Contains configuration constants such as DEVICE, EPOCHS, and LEARNING_RATE.
- data_loader: Provides the `get_loaders` function to load training and testing datasets.
- model: Defines the `IRPredictor` model architecture.
- train: Contains the `train` function to train the model.
- test: Contains the `test` function to evaluate the model.
Functions:
- main(): The entry point of the script. It initializes data loaders, the model, loss function, 
    and optimizer. It trains the model for a specified number of epochs, evaluates it, and saves 
    the trained model in both `.pth` and `.onnx` formats.
Usage:
Run this script directly to train and test the IRPredictor model:
        python main.py
"""
from data_loader import get_loaders
from model import IRPredictor
from train import train
from test import test
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_loss(train_losses, test_losses, r2s):
                        plt.figure(figsize=(10, 5))
                        plt.plot(train_losses, label="Train Loss")
                        plt.plot(test_losses, label="Test Loss")
                        plt.plot(r2s, label="R^2")
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.title("Loss vs Epoch")
                        plt.legend()
                        plt.grid(True)
                        plt.show()
                        
def main():
    train_loader, test_loader = get_loaders()
    train_losses = []
    test_losses = []
    r2s = []
    model = IRPredictor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, epoch)
        print(f"Epoch {epoch} - Training Loss: {train_loss:.6f}")
        train_losses.append(train_loss)
        if(epoch % 25 == 0):
            test_loss, _, r2 = test(model, test_loader, criterion, plot=True)
        test_loss, _, r2 = test(model, test_loader, criterion)
        test_losses.append(test_loss)
        r2s.append(r2)
    torch.save(model.state_dict(), "IR_model.pth")
    # dummy_input = torch.randn(1, 7).to(DEVICE)
    # torch.onnx.export(model, dummy_input, "IR_model.onnx",
    #                   input_names=["input"], output_names=["output"],
    #                   dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    #                   opset_version=11)

    plot_loss(train_losses, test_losses, r2s)
if __name__ == "__main__":
    main()



#remove outliers
#import onnx
