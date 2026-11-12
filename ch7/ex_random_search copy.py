from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
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

# 앞 슬라이드(그리드 서치)의 파이프라인
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())

dist = {"kneighborsclassifier__n_neighbors": randint(1, 101),   # 1~100 중 무작위
        "kneighborsclassifier__weights": ["uniform", "distance"]}

rs = RandomizedSearchCV(pipe, dist, n_iter=20, cv=5, random_state=42)  # 20개 조합만
rs.fit(X_train, y_train)                  # pipe: 앞 슬라이드의 파이프라인

print(rs.best_params_)
print(round(rs.best_score_, 3), round(rs.score(X_test, y_test), 3))
# {'kneighborsclassifier__n_neighbors': 33, 'kneighborsclassifier__weights': 'distance'}
# 0.802 0.81
