import numpy as np
from sklearn.linear_model import LinearRegression

# 자동차 무게(1,000파운드)와 연비(mpg) 7건
X = np.array([[3.5], [3.69], [3.44], [3.43], [4.34], [4.42], [2.37]])
y = np.array([18, 15, 18, 16, 15, 14, 24])

model = LinearRegression()   # 1. 모델 생성
model.fit(X, y)              # 2. 학습

print("b  =", round(model.intercept_, 2))   # 편향
print("w1 =", round(model.coef_[0], 2))     # 가중치
print("4,000파운드 예측 연비 =", round(model.predict([[4.0]])[0], 1))
# b  = 33.59
# w1 = -4.57
# 4,000파운드 예측 연비 = 15.3
