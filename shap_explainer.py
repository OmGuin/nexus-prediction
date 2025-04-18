import shap
import torch
import numpy as np
from model import IRPredictor
from config import DEVICE

def get_shap_values(model, background_data, input_tensor):
    model.eval()

    def model_forward(x):
        return model(torch.tensor(x, dtype = torch.float32).to(DEVICE)).detach().cpu().numpy()
    
    explainer = shap.Explainer(model_forward, background_data)
    shap_values = explainer(input_tensor)
    return shap_values
