import matplotlib.pyplot as plt
from utils import load_data
from config import Config
import numpy as np

X_train, X_val, y_train, y_val, scaler = load_data(Config.CSV_PATH, "HOMA-IR")



plt.figure(figsize=(12,4))
plt.subplot(121)
plt.hist(y_train, bins=30)
plt.title('Target Distribution')
plt.subplot(122)
plt.hist(np.log1p(y_train), bins=30)
plt.title('Log-Transformed Target')
plt.show()