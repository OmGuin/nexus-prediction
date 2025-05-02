import torch
import torch.nn as nn
import joblib

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
    
x_scaler = joblib.load('lin_model/x_scaler.pkl')
y_scaler = joblib.load('lin_model/y_scaler.pkl')
#Age, Gender, BMI, Body weight, Height
input_data = [[40, 1, 30.038349, 175.0,	64.00]]
input_data = x_scaler.transform(input_data)


input_tensor = torch.tensor(input_data, dtype=torch.float32)



model = RegressionNN()
model.load_state_dict(torch.load('lin_model/model.pth'))
model.eval() 



output = model(input_tensor)
output = y_scaler.inverse_transform(output.detach().numpy())
output = torch.tensor(output, dtype=torch.float32)
print(output.item())