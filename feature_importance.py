import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from data import load_data
import shap
import pandas as pd

def getGraph(input):
    X_train, _, _, _ = load_data()
    features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
       'BMI', 'Diabetes \nPedigree \nFunction', 'Age']
    input_df = pd.DataFrame([input], columns = features)
    with open('trained_xgb.pkl', 'rb') as file:
        model = pickle.load(file)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(input_df)

    fig, ax = plt.subplots(figsize=(8,4))
    shap.plots.waterfall(shap_values[0], show = False)
 
    plt.subplots_adjust(left = 0.30, right = 0.95, top = 0.9, bottom = 0.1)

    return fig




if __name__ == "__main__":
    print(type(getGraph([0, 129, 110, 46, 130, 67.1, 0.32, 30])))