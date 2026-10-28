import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([[3.5], [3.69], [3.44], [3.43], [4.34], [4.42], [2.37]])
y = np.array([18, 15, 18, 16, 15, 14, 24])

model = LinearRegression()   # 1. 모델 생성
model.fit(X, y)              # 2. 학습

print("b  =", round(model.intercept_, 2)) # 편향
print("w1 =", round(model.coef_[0], 2)) # 가중치
print("4,000파운드 예측 연비 =", round(model.predict([[4.0]])[0], 1))

plt.scatter(X, y, label="data") # 데이터(점)
plt.plot(X, model.predict(X), color="red", label="model")  # 학습된 직선
plt.xlabel("Weight (1,000 lb)")
plt.ylabel("MPG")
plt.legend()
plt.show()
