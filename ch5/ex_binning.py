import pandas as pd

titanic = pd.read_csv('dataset/titanic.csv')
df = titanic.copy()
df['연령대'] = pd.cut(df['Age'], bins=[0, 20, 40, 60, 80], labels=['20세 이하', '21~40세', '41~60세', '61~80세'])
df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Age'])

summary = (df.groupby(['Pclass', '연령대'], observed=True)['Survived'].agg(인원='count', 생존율='mean').round(2))
summary.loc[summary['인원'] < 5, '생존율'] = None # 작은 집단은 숨김
print(summary)
