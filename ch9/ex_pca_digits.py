import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

X = load_digits().data # (1797, 64)

pca = PCA().fit(X) # 주성분 64개 모두 계산
cum = np.cumsum(pca.explained_variance_ratio_) # 누적 합

print(cum[[1, 9, 19, 29]].round(3)) # 주성분 2 · 10 · 20 · 30개
print(PCA(n_components=0.95).fit(X).n_components_) # 0~1 실수: 비율로 지정
# [0.285 0.738 0.894 0.959]
# 29
