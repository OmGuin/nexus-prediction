
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = pd.read_csv('lin_model/bio copy.csv')
gender = {
    "M":1,
    "F":0
}
data["Gender"] = data["Gender"].map(gender)
data.dropna(inplace=True)
X = data[["Age","Gender","BMI","Body weight ","Height "]].values
y = data["Insulin "].values.reshape(-1, 1)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Perform PCA to reduce X to 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with y as the color
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y.flatten(), cmap='viridis', alpha=0.8)

# Add color bar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('y (Insulin)')

# Set axis labels
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')

# Set title
plt.title('3D PCA Visualization with y as Color')
plt.show()