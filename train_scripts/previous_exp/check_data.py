import pandas as pd
df = pd.read_parquet("/data/shuozhe/saved_dataset/math-500/test-00000-of-00001_verl.parquet")
# print(df)
print(df.columns)
print(df.iloc[0]["data_source"])
print(df.iloc[0]["reward_model"])
print(df.iloc[0]["prompt"][:1])


# from datasets import load_dataset
# ds = load_dataset("parquet", data_files="/path/to/data.parquet", split="train")
# print(ds.column_names)
# print(ds[0].keys())