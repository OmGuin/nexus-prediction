import scipy
import numpy as np
import pickle
import pandas as pd
from config import Config

FEATURES = Config.FEATURES
tunable_features = Config.tunable_features
with open("trained_xgb.pkl", "rb") as file:
  model = pickle.load(file)

def objective_function(tunable_inputs, fixed_inputs):
  bp, st, insulin, dpf, age = fixed_inputs
  glucose, bmi = tunable_inputs 
  
  inputs = [glucose, bp, st, insulin, bmi, dpf, age]
  df = pd.DataFrame([inputs])
  df.astype(float)
  return model.predict(df)[0]


def optimize_score(inputs):
  fixed_inputs = dict()
  for key, value in inputs.items():
     if key in Config.nontunable_features:
        fixed_inputs[key] = value

  # x0
  initial_tunable_inputs = [inputs['Glucose'], inputs['BMI']]
  
  bounds = Config.bounds
  
  result = scipy.optimize.minimize(
      objective_function,
      x0=initial_tunable_inputs,
      args=(fixed_inputs,),
      bounds=bounds,
      method='L-BFGS-B'
  )
  
  # Optimized inputs
  optimized_tunable_inputs = result.x
  print("Optimized Tunable Inputs:", optimized_tunable_inputs)


  optimized_score = objective_function(optimized_tunable_inputs, fixed_inputs)
  return optimized_score, optimized_tunable_inputs
  


print(optimize_score(dict(Glucose = 96, BloodPressure=122, SkinThickness=30, Insulin=100, BMI=22, DiabetesPedigreeFunction=0.3, Age=42)))
