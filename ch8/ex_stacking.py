import pandas as pd
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# 6장 타이타닉 데이터 준비 (피쳐 4개)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df["Age"] = df["Age"].fillna(df["Age"].median())     # 결측치: 중앙값으로 채움
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # 문자 → 숫자
X = df[["Pclass", "Sex", "Age", "Fare"]]             # 피쳐 4개
y = df["Survived"]                                   # 레이블: 생존 1 / 사망 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

estimators = [("lr", LogisticRegression(max_iter=1000)),
              ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))),
              ("tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
              ("rf", RandomForestClassifier(n_estimators=100, random_state=42))]
stack = StackingClassifier(estimators, final_estimator=LogisticRegression(), cv=5)
cv = cross_val_score(stack, X_train, y_train, cv=5).mean()
stack.fit(X_train, y_train)
print(round(cv, 3), round(stack.score(X_test, y_test), 3))
print(stack.final_estimator_.coef_.round(2))  # 메타 모델이 학습한 가중치
# 0.823 0.844
# [[2.   0.78 1.38 1.15]]
