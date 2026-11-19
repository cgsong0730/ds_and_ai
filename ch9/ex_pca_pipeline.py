from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

for n in [2, 5, 10, 20, 30]:
    pipe = make_pipeline(PCA(n_components=n), KNeighborsClassifier())
    cv = cross_val_score(pipe, X_train, y_train, cv=5).mean()
    pipe.fit(X_train, y_train)
    print(n, round(cv, 3), round(pipe.score(X_test, y_test), 3))
