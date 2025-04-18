class Config:
  FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
  FORMAL_FEATURES = ["Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree\nFunction", "Age"]
  tunable_features = ["Glucose", "BMI"]
  nontunable_features = ["BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction", "Age"]
  bounds = [(80, 130), (18, 40)]
  FULL_FEATURES = ["Pregnancies"] + FEATURES

    
    
