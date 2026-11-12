import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 6장 연비 데이터 준비 (특성 6개)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url).dropna()                       # 결측치 제거 (392행)
features = ["weight", "displacement", "acceleration",
            "cylinders", "horsepower", "model_year"]
X, y = df[features], df["mpg"]                       # 특성 6개 / 레이블 연비
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("학습 R2:", round(rf.score(X_train, y_train), 3))
print("평가 R2:", round(rf.score(X_test, y_test), 3))
# 학습 R2: 0.981
# 평가 R2: 0.888
