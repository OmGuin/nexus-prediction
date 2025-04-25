from flask import Flask, request, jsonify
from predict import load_model, predict
import torch
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from config import CSV_PATH

from config import DEVICE
from shap_explainer import get_shap_values

app = Flask(__name__)
model = load_model()


background_df = pd.read_csv("CSV_PATH").iloc[:, :-1]
background_data = background_df.sample(100).values

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        input_data = request.json["features"]  # list of 8 floats
        pred = predict(model, input_data)

        # SHAP
        shap_values = get_shap_values(model, background_data, [input_data])
        shap.plots.waterfall(shap_values[0], show=False)

        os.makedirs("static", exist_ok=True)
        shap_plot_path = "static/shap_plot.png"
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()

        return jsonify({
            "prediction": pred,
            "shap_plot": shap_plot_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/")
def index():
    return "Insulin Resistance Predictor API is running!"

if __name__ == "__main__":
    app.run(debug=True)
