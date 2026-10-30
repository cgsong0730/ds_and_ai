from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


iris = load_iris(as_frame=True)
X = iris.data # 특성(Feature): 꽃받침·꽃잎의 길이와 너비
y = iris.target # 레이블(Label): 품종 0, 1, 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dist = {"kneighborsclassifier__n_neighbors": randint(1, 101),   # 1~100 중 무작위
        "kneighborsclassifier__weights": ["uniform", "distance"]}

rs = RandomizedSearchCV(pipe, dist, n_iter=20, cv=5, random_state=42)  # 20개 조합만
rs.fit(X_train, y_train)                  # pipe: 앞 슬라이드의 파이프라인

print(rs.best_params_)
print(round(rs.best_score_, 3), round(rs.score(X_test, y_test), 3))
# {'kneighborsclassifier__n_neighbors': 33, 'kneighborsclassifier__weights': 'distance'}
# 0.802 0.81
