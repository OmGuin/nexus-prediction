import scipy
import numpy as np
import pickle
import pandas as pd
from config import Config

FEATURES = Config.features
tunable_features = Config.tunable_features

def model(x)
  return model.predict(x)[0]

def objective_function(tunable_inputs, fixed_inputs):
    bp, st, insulin, dpf, age = fixed_inputs
    glucose, bmi = tunable_inputs 
    
    inputs = [glucose, bp, st, insulin, bmi, dpf, age]
    df = pd.DataFrame(inputs, columns = Config.FEATURES)
    return model.predict(inputs)


def optimize_score(inputs):
  with open("trained_xgb.pkl", "rb") as file:
  model = pickle.load(file)

  
  fixed_inputs = {'age': 30}

  # x0
  initial_tunable_inputs = [inputs['Glucose'], inputs['BMI']]
  
  bounds = Config.bounds
  
  result = minimize(
      objective_function,
      x0=initial_tunable_inputs,
      args=(fixed_inputs,),
      bounds=bounds,
      method='L-BFGS-B'  # A good choice for bounded problems
  )
  
  # Optimized inputs
  optimized_tunable_inputs = result.x
  print("Optimized Tunable Inputs:", optimized_tunable_inputs)
  
  final_irscore = irscore_model([fixed_inputs['age']] + list(optimized_tunable_inputs))
  print("Final IRscore:", final_irscore)
    
  
  
  
  
