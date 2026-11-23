from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR.parent / "dataset" / "auto-mpg.data"
MODEL_PATH = SCRIPT_DIR.parent / "models" / "auto_mpg_model.joblib"


columns = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "name",
]

df = pd.read_csv(
    DATASET_PATH,
    sep=r"\s+",
    names=columns,
    na_values="?",
    quotechar='"',
)

# 전처리
numeric_columns = columns[:-1]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=numeric_columns)

features = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]
X = df[features]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions) ** 0.5

print("평균 제곱근 오차(RMSE):", round(rmse, 2))
print("결정계수(R²):", round(r2_score(y_test, predictions), 2))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
