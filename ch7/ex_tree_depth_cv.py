from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 6장 타이타닉 데이터 준비 (피쳐 4개)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df["Age"] = df["Age"].fillna(df["Age"].median())     # 결측치: 중앙값으로 채움
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # 문자 → 숫자
X = df[["Pclass", "Sex", "Age", "Fare"]]             # 피쳐 4개
y = df["Survived"]                                   # 레이블: 생존 1 / 사망 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

for d in [1, 2, 3, 4, 5, 7, 10, None]:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    cv = cross_val_score(tree, X_train, y_train, cv=5).mean()
    tree.fit(X_train, y_train)
    print(d, round(tree.score(X_train, y_train), 3), round(cv, 3), tree.get_n_leaves())
# 1 0.789 0.789 2
# 2 0.805 0.792 4
# 3 0.826 0.813 8
# 4 0.831 0.816 15
# 5 0.865 0.816 24
# 7 0.882 0.802 45
# 10 0.924 0.791 77
# None 0.98 0.763 168
