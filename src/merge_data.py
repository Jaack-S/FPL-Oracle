import os
import pandas as pd


output_file = "data/merged_data.csv"
input_dir = "data/raw/vaastav"

 # loop through files in inputdir
dfs = []
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv(output_file, index=False)

# merge files
if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print("Success")
else:
    print("Fail")    