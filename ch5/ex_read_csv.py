import pandas as pd

df = pd.read_csv('dataset/subway_congestion_level.csv', encoding='cp949') # 한글 공공데이터
print(df.shape)

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url) # 인터넷 주소도 바로 읽기
print(titanic.shape)
print(titanic.columns.tolist())
