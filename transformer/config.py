import torch
class Config:
    CSV_PATH = 'transformer/subsample.csv'
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    INPUT_DIM = 10  # Number of features in the dataset
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    DROPOUT = 0.1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
