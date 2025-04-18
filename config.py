import torch

CSV_PATH = "data.csv"
FEATURES = []

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
