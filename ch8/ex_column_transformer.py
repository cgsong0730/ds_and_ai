import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url) # 전처리하지 않은 원본
num_cols = ["Age", "SibSp", "Parch", "Fare"] # 숫자형 피쳐
cat_cols = ["Pclass", "Sex", "Embarked"] # 범주형 피쳐
X, y = df[num_cols + cat_cols], df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                         OneHotEncoder(handle_unknown="ignore"))
preprocess = ColumnTransformer([("num", num_pipe, num_cols),
                                ("cat", cat_pipe, cat_cols)])
model = Pipeline([("preprocess", preprocess),
                  ("model", LogisticRegression(max_iter=1000))])
cv = cross_val_score(model, X_train, y_train, cv=5).mean()
model.fit(X_train, y_train)
print(round(cv, 3), round(model.score(X_test, y_test), 3))
# 0.798 0.804
