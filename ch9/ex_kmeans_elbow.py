from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = load_iris(as_frame=True).data
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    print(k, round(km.inertia_, 1))
# 1 681.4
# 2 152.3
# 3 78.9 ← 감소 폭이 이 지점부터 크게 줄어듦
# 4 57.4
# 5 46.5
# …
# 10 28.5

for k in range(2, 7):
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
    print(k, round(silhouette_score(X, labels), 3))
