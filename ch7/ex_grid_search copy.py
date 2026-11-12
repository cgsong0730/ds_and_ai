from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 6장 타이타닉 데이터 준비 (피쳐 4개)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df["Age"] = df["Age"].fillna(df["Age"].median())     # 결측치: 중앙값으로 채움
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # 문자 → 숫자
X = df[["Pclass", "Sex", "Age", "Fare"]]             # 피쳐 4개
y = df["Survived"]                                   # 레이블: 생존 1 / 사망 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
params = {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 15, 25, 51],
          "kneighborsclassifier__weights": ["uniform", "distance"]}

grid = GridSearchCV(pipe, params, cv=5) # 8 x 2 = 16개 조합 x 5겹
grid.fit(X_train, y_train)

print(grid.best_params_)
print("교차 검증 최고 점수:", round(grid.best_score_, 3))
print("평가 정확도:", round(grid.score(X_test, y_test), 3))
# {'kneighborsclassifier__n_neighbors': 3, 'kneighborsclassifier__weights': 'uniform'}
# 교차 검증 최고 점수: 0.812
# 평가 정확도: 0.816
