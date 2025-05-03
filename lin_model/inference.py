import torch
import torch.nn as nn
import joblib

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
    x_scaler = joblib.load('lin_model/x_scaler.pkl')
    y_scaler = joblib.load('lin_model/y_scaler.pkl')
    model = RegressionNN()
    model.load_state_dict(torch.load('lin_model/IR_MLP.pth'))
    model.eval()

    x_scaler = joblib.load('lin_model/x_scaler.pkl')
    y_scaler = joblib.load('lin_model/y_scaler.pkl')

    input_data = x_scaler.transform(input_data)

    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    output = model(input_tensor)
    
    output = y_scaler.inverse_transform(output.detach().numpy())
    output = convert(output.item())

    return output.item()



input_data = [[40, 1, 30.038349, 175.0,	64.00]]

print(calculate_irscore(input_data))



