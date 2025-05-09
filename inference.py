import numpy as np

def calculate_irscore(x_new):
    beta = np.load("lin_model/linear_regression_beta.npy")
    x_new_bias = np.hstack([np.ones((x_new.shape[0], 1)), x_new])
    y_pred = x_new_bias @ beta
    convert = lambda x: 100/(1+np.exp(1.5*x-3))
    return convert(y_pred[0,0])







