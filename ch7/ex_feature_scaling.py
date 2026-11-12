import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 시험 점수 데이터
X = np.array([[60], [70], [80]])

# 표준화
standard = StandardScaler()
X_std = standard.fit_transform(X)

# 최소-최대 정규화
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)

print("표준화:", X_std.round(2).ravel())
print("정규화:", X_minmax.ravel())
# 표준화: [-1.22  0.    1.22]
# 정규화: [0.  0.5 1. ]
