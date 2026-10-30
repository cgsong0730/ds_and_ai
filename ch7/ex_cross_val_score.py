from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris = load_iris(as_frame=True)
X = iris.data # 특성(Feature): 꽃받침·꽃잎의 길이와 너비
y = iris.target # 레이블(Label): 품종 0, 1, 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
scores = cross_val_score(model, X_train, y_train, cv=5)

print(scores.round(3))
print("평균:", round(scores.mean(), 3))
# [0.917 1.    0.917 0.958 1.   ]
# 평균: 0.958
