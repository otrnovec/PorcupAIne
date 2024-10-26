from pathlib import Path
import pandas as pd
from preprocess_data import change_column_types, split_train_val_test

df = pd.read_csv(Path(__file__).parent / "data" / "paro_preprocessed.csv")
df = change_column_types(df)
df_train, df_val, df_test = split_train_val_test(df)

print(df.info())
