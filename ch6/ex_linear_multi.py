import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url).dropna()          # 결측치 제거 (392행)

features = ["weight", "displacement", "acceleration",
            "cylinders", "horsepower", "model_year"]
X, y = df[features], df["mpg"]          # 특성 6개 / 레이블 연비

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
print("특성 6개 R2:", round(model.score(X_test, y_test), 3))
# 특성 6개 R2: 0.794

one = LinearRegression().fit(X_train[["weight"]], y_train)
print("무게만   R2:", round(one.score(X_test[["weight"]], y_test), 3))
# 무게만   R2: 0.653
