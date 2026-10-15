import pandas as pd
df = pd.DataFrame({'측정소': ['종로구', '강남구'], '미세먼지': [32, 41]})

df.to_csv('a.csv') # 행 번호까지 저장
print(pd.read_csv('a.csv'))

df.to_csv('b.csv', index=False) # 행 번호 빼고 저장
print(pd.read_csv('b.csv'))
