import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.model_selection import train_test_split

# 6장 연비 데이터 준비 (특성 6개)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url).dropna()                       # 결측치 제거 (392행)
features = ["weight", "displacement", "acceleration",
            "cylinders", "horsepower", "model_year"]
X, y = df[features], df["mpg"]                       # 특성 6개 / 레이블 연비
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

for model in [GradientBoostingRegressor(random_state=42),
              HistGradientBoostingRegressor(random_state=42)]:
    model.fit(X_train, y_train)
    print(type(model).__name__, round(model.score(X_train, y_train), 3),
          round(model.score(X_test, y_test), 3))
# GradientBoostingRegressor 0.97 0.873
# HistGradientBoostingRegressor 0.962 0.881
