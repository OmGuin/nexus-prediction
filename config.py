class Config:
  def __init__(self):
    self.FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    self.tunable_features = ["Glucose", "BMI"]
