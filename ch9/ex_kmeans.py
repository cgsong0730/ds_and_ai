from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd

iris = load_iris(as_frame=True)
X = iris.data # 피쳐 4개만 사용 (정답 y는 사용 안 함)

km = KMeans(n_clusters=3, random_state=42)
km.fit(X) # 비지도 학습: fit(X)

print(km.labels_[:5], km.labels_[50:55], km.labels_[100:105]) # 군집 결과
print(km.cluster_centers_.round(2)) # 군집 중심 좌표
print(round(km.inertia_, 2)) # 관성: 중심까지 거리 제곱의 합

# [1 1 1 1 1] [0 2 0 2 2] [0 2 0 0 0]
# [[6.85 3.08 5.72 2.05]
#  [5.01 3.43 1.46 0.25]
#  [5.88 2.74 4.39 1.43]]
# 78.86

print(pd.crosstab(iris.target_names[iris.target], km.labels_, rownames=["품종"], colnames=["군집"]))
