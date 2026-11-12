import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# 6장 타이타닉 데이터 준비 (피쳐 4개)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df["Age"] = df["Age"].fillna(df["Age"].median())     # 결측치: 중앙값으로 채움
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # 문자 → 숫자
X = df[["Pclass", "Sex", "Age", "Fare"]]             # 피쳐 4개
y = df["Survived"]                                   # 레이블: 생존 1 / 사망 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

params = {"learning_rate": [0.05, 0.1, 0.5, 1.0],
          "n_estimators": [50, 100, 200, 500]}
grid = GridSearchCV(GradientBoostingClassifier(random_state=42), params, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_, round(grid.best_score_, 3), round(grid.score(X_test, y_test), 3))
# {'learning_rate': 0.05, 'n_estimators': 500} 0.825 0.793
