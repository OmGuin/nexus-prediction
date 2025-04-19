import torch

CSV_PATH = "data.csv"
FEATURES = ['Gender', 'AgeYears', 'Race_Ethnicity', 'HouseholdSize', 'FamilySize', 'InterviewSampleWeight', 'ExamSampleWeight']

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
