import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

rf = RandomForestClassifier(n_estimators=100, oob_score=True,
                            random_state=42)
cv = cross_val_score(rf, X_train, y_train, cv=5).mean()
rf.fit(X_train, y_train)
print(round(rf.score(X_train, y_train), 3), round(cv, 3),
      round(rf.score(X_test, y_test), 3), round(rf.oob_score_, 3))
print(pd.Series(rf.feature_importances_, index=X.columns).round(3))
# 0.98 0.803 0.832 0.812
# Pclass    0.092
# Sex       0.290
# Age       0.279
# Fare      0.339
