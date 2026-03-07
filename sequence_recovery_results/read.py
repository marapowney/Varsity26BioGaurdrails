import pandas as pd

FILE = "GENERator-v2-eukaryote-1.2b-base_bfloat16.parquet"
df = pd.read_parquet(FILE)

print(df)