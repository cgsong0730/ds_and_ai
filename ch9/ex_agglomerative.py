import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

iris = load_iris(as_frame=True)
X = iris.data

for link in ["ward", "complete", "average", "single"]:
    agg = AgglomerativeClustering(n_clusters=3, linkage=link)
    labels = agg.fit_predict(X) # 학습 + 군집 번호 반환
    print(link, np.bincount(labels), round(adjusted_rand_score(iris.target, labels), 3))
