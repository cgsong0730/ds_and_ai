import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 자동차 무게(1,000파운드)와 연비(mpg) 7건
X = np.array([[3.5], [3.69], [3.44], [3.43], [4.34], [4.42], [2.37]])
y = np.array([18, 15, 18, 16, 15, 14, 24])

# 연비 데이터의 X_train, X_test, y_train, y_test 사용 (특성 6개)
for name, reg in [("선형", LinearRegression()), ("릿지", Ridge(alpha=1.0)), ("라쏘", Lasso(alpha=1.0))]:
    model = make_pipeline(StandardScaler(), reg).fit(X_train, y_train)
    print(name, round(model.score(X_test, y_test), 3), model[-1].coef_.round(2))
