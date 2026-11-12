from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

# 6장 연비 데이터 준비 (특성 6개)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url).dropna()                       # 결측치 제거 (392행)
features = ["weight", "displacement", "acceleration",
            "cylinders", "horsepower", "model_year"]
X, y = df[features], df["mpg"]                       # 특성 6개 / 레이블 연비
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 연비 데이터의 X_train, X_test, y_train, y_test 사용 (특성 6개)
for d in [1, 2, 3, 5, 8, None]:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    print(d, round(tree.score(X_train, y_train), 3),
             round(tree.score(X_test, y_test), 3), tree.get_n_leaves())
# 1 0.592 0.489 2
# 2 0.757 0.583 4
# 3 0.84 0.733 8
# 5 0.931 0.772 29
# 8 0.982 0.804 131
# None 1.0 0.755 261
