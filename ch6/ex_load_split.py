from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)
X = iris.data                  # 특성(Feature): 꽃받침·꽃잎의 길이와 너비
y = iris.target                # 레이블(Label): 품종 0, 1, 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(X.shape, X_train.shape, X_test.shape)
# (150, 4) (120, 4) (30, 4)
