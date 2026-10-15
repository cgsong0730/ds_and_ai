import pandas as pd

titanic = pd.read_csv('dataset/titanic.csv')
df = titanic.copy()

df['호칭'] = df['Name'].str.extract(r',\s*([^.]+)\.') # Mr, Mrs, ...
df['이름'] = df['Name'].str[0] + '**' # 성의 첫 글자만 남김
df['티켓'] = df['Ticket'].str[:2] + '****' # 앞 2자리만 남김
df = df.drop(columns=['Name', 'Ticket', 'Cabin']) # 원본 식별 열 삭제

print(df[['PassengerId', '이름', '호칭', '티켓', 'Age']].head(3))
