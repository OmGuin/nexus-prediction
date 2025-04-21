import pyreadstat
import os

for file_name in os.listdir("data_preprocessing/xpts"):
    df, meta = pyreadstat.read_xport(f'data_preprocessing/xpts/{file_name}')
    df.to_csv(f'data_preprocessing/csvs/{file_name.removesuffix(".xpt")}.csv', index=False)
    print(file_name, "convereted to csv")

# print(meta.column_labels)
# print(df.columns)
# print(df.head())
# print(df.describe())

