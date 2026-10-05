import pandas as pd
import numpy as np

# Titanic 데이터셋 불러오기
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 중앙값으로 대체

df['Age'] = df['Age'].fillna(df['Age'].median())
