from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

df = pd.read_csv("transformer/stuff copy.csv")
x = df.drop(columns=["HOMA-IR"])
y = df["HOMA-IR"]
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['pc 1', 'pc 2', 'pc 3'])
principalDf['HOMA-IR'] = y.values


# 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# You can color points by the target variable "HOMA-IR" if you want
sc = ax.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2], 
                c=df['HOMA-IR'], cmap='viridis', s=50)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA Visualization')

plt.colorbar(sc, label='HOMA-IR')

plt.show()
