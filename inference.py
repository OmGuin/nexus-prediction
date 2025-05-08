import torch
import torch.nn as nn
import pickle
import shap
import numpy as np


def convert(x):
    return 100/(x+1)

class RegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.model(x)

def calculate_irscore(input_data): #Age (years), Gender (1 = M, 0 = F), BMI (kg/m^2), Body weight (pounds), Height (inches)
    with open('lin_model/x_scaler.pkl', 'rb') as f:
        x_scaler = pickle.load(f)
    with open('lin_model/y_scaler.pkl', 'rb') as f:
        y_scaler = pickle.load(f)

    model = RegressionNN()
    model.load_state_dict(torch.load('lin_model/IR_MLP.pth'))
    model.eval()

    input_data = x_scaler.transform(np.array(input_data))
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)
    output = y_scaler.inverse_transform(output.numpy())
    with open('lin_model/background.pkl', 'rb') as f:
        background_data = pickle.load(f)
    background = torch.tensor(background_data, dtype=torch.float32)
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(input_tensor)
    output = convert(output.item())
    

    return output, shap_values[0]


"""
input_data = [[40, 1, 30.038349, 175.0,	64.00]]

ir_score, shap_vals = calculate_irscore(input_data)
feature_names = ['Age', 'Gender', "BMI", "Body weight", "Height"]

print(f"IR Score: {ir_score:.2f}")
print("Feature importances (SHAP values):")
for name, val in zip(feature_names, shap_vals):
    print(f"{name}: {val[0]:.4f}")


"""