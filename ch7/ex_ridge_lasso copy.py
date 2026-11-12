from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 6장 연비 데이터 준비 (특성 6개)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url).dropna()                       # 결측치 제거 (392행)
features = ["weight", "displacement", "acceleration",
            "cylinders", "horsepower", "model_year"]
X, y = df[features], df["mpg"]                       # 특성 6개 / 레이블 연비
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 선형 회귀 (규제 없음, 비교 기준)
# ---------------------------------------------------------
linear_model = make_pipeline(StandardScaler(), LinearRegression())
linear_model.fit(X_train, y_train)
linear_score = linear_model.score(X_test, y_test)
linear_coef = linear_model[-1].coef_
print("선형", round(linear_score, 3), linear_coef.round(2))

# ---------------------------------------------------------
# 릿지 회귀 (L2 규제)
# ---------------------------------------------------------
ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge_model.fit(X_train, y_train)
ridge_score = ridge_model.score(X_test, y_test)
ridge_coef = ridge_model[-1].coef_
print("릿지", round(ridge_score, 3), ridge_coef.round(2))

# ---------------------------------------------------------
# 라쏘 회귀 (L1 규제)
# ---------------------------------------------------------
lasso_model = make_pipeline(StandardScaler(), Lasso(alpha=1.0))
lasso_model.fit(X_train, y_train)
lasso_score = lasso_model.score(X_test, y_test)
lasso_coef = lasso_model[-1].coef_
print("라쏘", round(lasso_score, 3), lasso_coef.round(2))

# 가중치 순서: weight, displacement, acceleration, cylinders, horsepower, model_year
# 선형 0.794 [-5.51  0.1   0.17 -0.2  -0.09  2.76]
# 릿지 0.794 [-5.34 -0.01  0.13 -0.21 -0.17  2.74]
# 라쏘 0.802 [-4.67 -0.16  0.   -0.   -0.19  2.01]
