import torch

CSV_PATH = "data1.csv"
#FEATURES = ['Gender', 'AgeYears', 'Race_Ethnicity', 'HouseholdSize', 'FamilySize', 'InterviewSampleWeight', 'ExamSampleWeight']
FEATURES = ['principal component 1', 'principal component 2', 'principal component 3']
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
