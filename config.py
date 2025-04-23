import torch

CSV_PATH = "data_preprocessing/subsample.csv"
with open("data_preprocessing/num_features.txt", "r") as f:
    num_features = int(f.read())
FEATURES = ["X"] * num_features
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
