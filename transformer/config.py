import torch
class Config:
    CSV_PATH = 'transformer/stuff.csv'
    BATCH_SIZE = 8
    EPOCHS = 200
    LEARNING_RATE = 0.0001
    INPUT_DIM = 5
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    DROPOUT = 0.1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    WEIGHT_DECAY = 1e-4
