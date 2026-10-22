from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)
df["Age"] = df["Age"].fillna(df["Age"].median()) # 결측치: 중앙값으로 채움
df["Sex"] = df["Sex"].map({"male": 0, "female": 1}) # 문자 → 숫자
X = df[["Pclass", "Sex", "Age", "Fare"]] # 특성 4개
y = df["Survived"] # 레이블: 생존 1 / 사망 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("정확도:", round(clf.score(X_test, y_test), 3))
# 정확도: 0.788
print(clf.predict_proba(X_test[:1]).round(3)) # [사망 확률, 생존 확률]
# [[0.894 0.106]]

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))