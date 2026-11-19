import pandas as pd
from sklearn.ensemble import IsolationForest

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url).dropna() # 자동차 392대
features = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration"]

iso = IsolationForest(contamination=0.02, random_state=42)
df["pred"] = iso.fit_predict(df[features]) # 이상 판정: 8대
df["score"] = iso.decision_function(df[features])
cols = ["name", "mpg", "cylinders", "horsepower", "weight", "score"]

print(df.sort_values("score")[cols].head(5).round(3)) # 점수가 낮은 5대
