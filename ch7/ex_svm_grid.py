from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 6장 타이타닉 데이터 준비 (피쳐 4개)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df["Age"] = df["Age"].fillna(df["Age"].median())     # 결측치: 중앙값으로 채움
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # 문자 → 숫자
X = df[["Pclass", "Sex", "Age", "Fare"]]             # 피쳐 4개
y = df["Survived"]                                   # 레이블: 생존 1 / 사망 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipe = make_pipeline(StandardScaler(), SVC())
params = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, params, cv=5).fit(X_train, y_train)
print(grid.best_params_, round(grid.best_score_, 3), round(grid.score(X_test, y_test), 3))
# {'svc__C': 100, 'svc__gamma': 0.1} 0.825 0.816
