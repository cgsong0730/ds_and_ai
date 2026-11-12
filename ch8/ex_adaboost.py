import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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

stump = DecisionTreeClassifier(max_depth=1)  # 약한 학습기: 그루터기
for n in [10, 50, 100, 200]:
    ada = AdaBoostClassifier(stump, n_estimators=n, random_state=42)
    cv = cross_val_score(ada, X_train, y_train, cv=5).mean()
    ada.fit(X_train, y_train)
    print(n, round(ada.score(X_train, y_train), 3), round(cv, 3),
          round(ada.score(X_test, y_test), 3))
# 10 0.801 0.795 0.777
# 50 0.813 0.791 0.76
# 100 0.817 0.802 0.754
# 200 0.826 0.809 0.754
