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
  print(fixed_inputs)
  print(tunable_inputs)
  bp, st, insulin, dpf, age = fixed_inputs
  glucose, bmi = tunable_inputs 
  inputs = [0, glucose, bp, st, insulin, bmi, dpf, age]
  print(inputs)
  df = pd.DataFrame([inputs], columns = ['Pregnancies'] + FEATURES)
  x = model.predict(df)[0]
  print(x)
  return x
  #return model.predict(df)[0]


def optimize_score(inputs):
  fixed_inputs = []
  for key, value in inputs.items():
     if key in Config.nontunable_features:
        fixed_inputs.append(value)

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
  print(result.x)

  #optimized_score = objective_function(optimized_tunable_inputs, fixed_inputs)
  #return optimized_score, optimized_tunable_inputs
  

optimize_score(dict(Glucose = 96, BloodPressure=122, SkinThickness=30, Insulin=100, BMI=22, DiabetesPedigreeFunction=0.3, Age=42))
#print(optimize_score(dict(Glucose = 96, BloodPressure=122, SkinThickness=30, Insulin=100, BMI=22, DiabetesPedigreeFunction=0.3, Age=42)))
