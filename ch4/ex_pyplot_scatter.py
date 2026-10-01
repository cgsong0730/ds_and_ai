import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0) # seed()는 그 난수 생성기의 시작점을 정해주는 함수, 0으로 고정 
x = np.random.rand(50) # 랜덤한 값 50개

# y는 “대략 2x에 가까운 값”을 만들되, 작은 랜덤 노이즈를 섞어주는 식
y = 2 * x + np.random.randn(50) * 0.3

plt.scatter(x, y, color='steelblue', alpha=0.7, edgecolors='black') # 2개의 피처를 요구함
plt.xlabel('x')
plt.ylabel('y')
plt.title('Basic Scatter')
plt.show()
