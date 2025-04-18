import torch
from model import IRPredictor
from config import DEVICE

def load_model(path = "IR_model.pth"):
    model = IRPredictor().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def predict(model, input_list):
    with torch.no_grad():
        input_tensor = torch.tensor(input_list, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
        return output.item()