from datasets import load_dataset
import pandas as pd

ds = load_dataset("d0r1h/customer_churn")

df = ds['train'].to_pandas()


print(df.head())