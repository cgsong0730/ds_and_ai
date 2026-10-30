from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)
X = iris.data # 특성(Feature): 꽃받침·꽃잎의 길이와 너비
y = iris.target # 레이블(Label): 품종 0, 1, 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
params = {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 15, 25, 51], "kneighborsclassifier__weights": ["uniform", "distance"]}

grid = GridSearchCV(pipe, params, cv=5) # 8 x 2 = 16개 조합 x 5겹
grid.fit(X_train, y_train)

print(grid.best_params_)
print("교차 검증 최고 점수:", round(grid.best_score_, 3))
print("평가 정확도:", round(grid.score(X_test, y_test), 3))
# {'kneighborsclassifier__n_neighbors': 3, 'kneighborsclassifier__weights': 'uniform'}
# 교차 검증 최고 점수: 0.812
# 평가 정확도: 0.816
