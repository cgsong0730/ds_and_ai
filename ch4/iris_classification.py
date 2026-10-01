import pandas as pd
import numpy as np
import mglearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris_dataset = load_iris()

print("iris_dataset's key:", iris_dataset.keys())
# print("iris_dataset's key:", type(iris_dataset.keys()))

# 출력
# print(iris_dataset['data'])
# print(iris_dataset['target'])
# print(iris_dataset['frame'])
# print(iris_dataset['target_names'])
# print(iris_dataset['DESCR'])
print(iris_dataset['feature_names'])
# print(iris_dataset['filename'])

# print(type(iris_dataset['data']))
print(iris_dataset['data'][:5])
# print(iris_dataset['data'].shape)

# train은 학습 데이터, test는 성능 분석 데이터, X는 data, y는 target
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

# 출력
# print("X_train: \n", X_train)
# print(X_train.shape)
# print("y_train: \n", y_train)
# print(y_train.shape)

# print("X_test: \n", X_test)
# print(y_test.shape)
# print("y_test", y_test)
# print(y_test.shape)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(15, 15),
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=.8,
    cmap=mglearn.cm3
)
plt.show()

# 훈련 데이터로 KNN 모델을 학습
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# 새로운 데이터로 target 예측
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("prediction:", prediction)
print("target name:", iris_dataset['target_names'][prediction])

# 테스트 데이터로 target 예측
y_prediction = knn.predict(X_test)
print("y_prediction: ", y_prediction)

# print("testset's accuracy: {:.2f}".format(np.mean(y_prediction == y_test)))

# 테스트용 target와 예측한 target을 비교하여 정답 비율 분석
print("testset's accuracy: {:.2f}".format(knn.score(X_test, y_test)))
print(np.mean(y_prediction == y_test))
