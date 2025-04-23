import torch
from config import DEVICE

def train(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        #print(y)

        optimizer.zero_grad()
        output = model(X)

        #print(output)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    return running_loss / len(loader)