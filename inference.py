import numpy as np



def calculate_irscore(x_new):
    beta = np.load("lin_model/linear_regression_beta.npy")
    x_new_bias = np.hstack([np.ones((x_new.shape[0], 1)), x_new])
    y_pred = x_new_bias @ beta

    # Convert predicted HOMA-IR to IRScore
    convert = lambda x: 100 / (1 + np.exp(1.5 * x - 3))
    ir_score = convert(y_pred[0, 0])

    # Feature contributions (excluding bias)
    feature_names = ['Age', 'Gender', 'BMI', 'Body weight', 'Height']
    contributions = x_new[0] * beta[1:].flatten()

    contribution_dict = dict(zip(feature_names, contributions))

    return ir_score, contribution_dict



