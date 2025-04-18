from config import DEVICE, EPOCHS, LEARNING_RATE
from data_loader import get_loaders
from model import IRPredictor
from train import train
from test import test
import torch
import torch.optim as optim
import torch.nn as nn

def main():
    train_loader, test_loader = get_loaders()
    model = IRPredictor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, epoch)
        print(f"Epoch {epoch} - Training Loss: {train_loss:.6f}")
        test(model, test_loader, criterion)
    torch.save(model.state_dict(), "IR_model.pth")
    dummy_input = torch.randn(1, 8).to(DEVICE)
    torch.onnx.export(model, dummy_input, "IR_model.onnx",
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=11)


if __name__ == "__main__":
    main()
