import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(42)
X_normal = rng.normal(0, 1, size=(300, 2)) # 정상 데이터 300개
X_out = rng.uniform(-6, 6, size=(15, 2)) # 이상 데이터 15개
X = np.vstack([X_normal, X_out])

iso = IsolationForest(contamination=0.05, random_state=42)
pred = iso.fit_predict(X) # 정상 1, 이상 -1

print(np.unique(pred, return_counts=True))
print(iso.decision_function(X[[0, -1]]).round(3))
# (array([-1,  1]), array([ 16, 299]))
# [ 0.191 -0.132]
