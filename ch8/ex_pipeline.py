import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
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

pipe = Pipeline([("scaler", StandardScaler()),     # (단계 이름, 객체)
                 ("svc", SVC(C=100, gamma=0.1))])
pipe.fit(X_train, y_train)                         # 스케일링 → 학습
print(round(pipe.score(X_test, y_test), 3))        # 스케일링 → 예측 → 평가
print(pipe.named_steps["scaler"].mean_.round(2))   # 학습 데이터의 평균
print(make_pipeline(StandardScaler(), SVC()).named_steps.keys())
# 0.816
# [ 2.31  0.36 29.46 31.82]
# dict_keys(['standardscaler', 'svc'])
