import pandas as pd
import numpy as np

# Titanic 데이터셋 불러오기
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 컬럼 자체를 삭제
df = df.drop('Cabin', axis=1)

# 해당 행만 삭제
df = df.dropna(subset=['Embarked'])

# 결측치가 하나라도 있는 행 전체를 삭제
df = df.dropna()