import pandas as pd

df = pd.read_csv('dataset/titanic.csv', encoding='cp949')

high = df[df['Age'] >= 40]
print(high)
print(df.groupby('Pclass')['Age'].mean().round(1))
