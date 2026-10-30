import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
X_train = df[["Pclass", "Sex", "Age", "Fare"]]
y_train = df["Survived"]

# 타이타닉 데이터의 X_train, y_train을 다시 학습용 / 검증용으로 분할
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

for k in [1, 3, 5, 7, 15, 51]:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    model.fit(X_tr, y_tr)
    print(k, round(model.score(X_tr, y_tr), 3), round(model.score(X_val, y_val), 3))
