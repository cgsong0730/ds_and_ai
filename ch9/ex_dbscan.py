import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

X, y = make_moons(n_samples=300, noise=0.08, random_state=42)
rng = np.random.default_rng(42)
X = np.vstack([X, rng.uniform(-1.5, 2.5, size=(10, 2))]) # 동떨어진 점 10개 추가
km = KMeans(n_clusters=2, random_state=42).fit_predict(X)
db = DBSCAN(eps=0.2, min_samples=5).fit_predict(X) # 잡음은 -1

print(np.unique(db, return_counts=True))
# (array([-1,  0,  1]), array([  8, 151, 151]))
