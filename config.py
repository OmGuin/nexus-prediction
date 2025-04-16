class Config:
  def __init__(self):
    self.FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    self.tunable_features = ["Glucose", "BMI"]
    self.nontunable_features = ["BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction", "Age"]
    self.bounds = [(80, 130), (18, 40)]
    
