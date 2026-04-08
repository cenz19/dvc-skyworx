import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/data.csv")
df.drop("no_referensi", axis=1, inplace=True)
df.drop("kondisi_kartu", axis=1, inplace=True)

df["tgl_approval"] = pd.to_datetime(df["tgl_approval"])
df["target"] = df["kolek"].apply(lambda x: 0 if x == 1 else 1)
df.drop("kolek", axis=1, inplace=True)
df = df.sort_values("tgl_approval", ignore_index=True)
split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

train.drop("tgl_approval", axis=1, inplace=True)
test.drop("tgl_approval", axis=1, inplace=True)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
