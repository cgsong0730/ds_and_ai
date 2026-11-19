import pandas as pd
from sklearn.ensemble import VotingClassifier
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

lr = LogisticRegression(max_iter=1000)
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
estimators = [("lr", lr), ("knn", knn), ("tree", tree)]

for v in ["hard", "soft"]:
    vote = VotingClassifier(estimators, voting=v)
    cv = cross_val_score(vote, X_train, y_train, cv=5).mean()
    vote.fit(X_train, y_train)
    print(v, round(cv, 3), round(vote.score(X_test, y_test), 3))
# hard 0.817 0.832
# soft 0.823 0.827
