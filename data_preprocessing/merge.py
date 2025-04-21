import pandas as pd
import glob

desired_columns = ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'DMQMILIZ',
                   'DMQADFC', 'DMDBORN4', 'DMDCITZN', 'DMDYRSUS', 'DMDEDUC3',
                   'DMDEDUC2','DMDMARTL','RIDEXPRG','INDHHIN2','DMDHHSIZ',
                   'DMDFMSIZ','DMDHHSZA','DMDHHSZB','DMDHHSZE','DR1CCMTX',
                   'DR1FS','DR1_040Z','DRQSDIET','DRQSDT1','DRQSDT2','DRQSDT3',
                   'DRQSDT4','DRQSDT5','DRQSDT6', 'DRQSDT7', 'DRQSDT8',
                   'DRQSDT9','DRQSDT10', 'DRQSDT11','DRQSDT12','DRQSDT91',
                   'BMXWT','BMXHT','BMXWAIST','ENQ010','ENQ020','SPQ010','SPQ020',
                   'SPQ040','SPQ060','SPQ070d','SPQ070e','ENQ100', 'RHQ131', 'RHD143',
                   'ALQ101','ALQ110','ALQ130','ALQ141Q','ALQ151','AUQ154','BPQ020','BPQ080',
                   'CDQ010','CBD120','DIQ010','DIQ160','DIQ170','DIQ175N','ECD070A','MCQ080','MCQ092','MCQ220','MCQ300c','MCQ365c','MCQ365d',
                   'SMQ020', 'SMD030', 'SMQ040','SMD415', 'SMD415A', 'LBDGLUSI', 'LBXIN']  # example

csv_files = glob.glob("C:/Users/aguin/Desktop/NexFlow/nexus-prediction/data_preprocessing/csvs/*.csv")


merged_df = None

for file in csv_files:
    df = pd.read_csv(file)
    
    filtered = df[[col for col in desired_columns if col in df.columns]]
    
    if 'SEQN' not in filtered.columns:
        continue

    filtered.set_index('SEQN', inplace=True)
    
    if merged_df is None:
        merged_df = filtered
    else:
        # Add values, aligning by ID and column name
        merged_df = merged_df.add(filtered, fill_value=0)

final_df = merged_df.reset_index()

final_df.to_csv("merged_output.csv", index=False)
