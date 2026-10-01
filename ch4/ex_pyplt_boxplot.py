import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use("Agg")  # GUI 환경이 없는 서버/터미널에서 안전하게 동작

data = np.random.randn(100, 5) # 평균 0, 표준편차 1인 정규분포”에서 난수를 100행 × 5열 형태로 뽑아 배열로 만드는 
plt.boxplot(data)

plt.title("Boxplot Example")
plt.xlabel("Sample")
plt.ylabel("Value")
plt.tight_layout()

plt.show()
# plt.savefig("boxplot.png", dpi=150)
# print("Saved: boxplot.png")
