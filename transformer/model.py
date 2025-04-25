import torch.nn as nn

class TransformerRegression2(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # [batch_size, features] -> [batch_size, sequence_length=features, d_model]
        x = self.input_projection(x).unsqueeze(1) 
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.output_layer(x).squeeze(-1)




class GFTNetX(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.SiLU(),  # 
            nn.Dropout(dropout//2),
            nn.Linear(d_model//2, 1))
        
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)  # [B,1,D]
        x = self.encoder(x)
        return self.output(x.squeeze(1)).squeeze(-1)




class TransformerRegression1(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128,  # Added
            dropout=dropout,
            batch_first=True      # Important!
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        x = self.input_projection(x).unsqueeze(1)  # [batch, 1, d_model]
        x = self.transformer_encoder(x)
        x = x.squeeze(1)  # [batch, d_model]
        return self.output_layer(x).squeeze(-1)  # [batch]

class GFTNet(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)  # → [batch_size, num_features, d_model]
        x = self.transformer(x)        # → same shape
        x = x.mean(dim=1)              # Pool over feature tokens
        out = self.regressor(x)        # → [batch_size, 1]
        return out.squeeze(1)
