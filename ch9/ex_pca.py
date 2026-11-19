from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)

X_std = StandardScaler().fit_transform(iris.data) # 표준화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std) # 4차원 → 2차원

print(X_std.shape, "→", X_pca.shape)
print(pca.explained_variance_ratio_.round(3)) # 설명된 분산 비율
print(pca.components_.round(2)) # 주성분 = 피쳐의 가중합
# (150, 4) → (150, 2)
# [0.73  0.229]
# [[ 0.52 -0.27  0.58  0.56]
#  [ 0.38  0.92  0.02  0.07]]
