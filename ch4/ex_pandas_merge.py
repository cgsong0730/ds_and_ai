import pandas as pd

df1 = pd.DataFrame({'지점': ['강남', '수원', '부산'],
                    '매출': [195000, 250500, 252000]})
df2 = pd.DataFrame({'지점': ['강남', '수원', '대전'],
                    '지역': ['서울', '경기', '대전']})

print(df1)
print(df2)
print(pd.merge(df1, df2, on='지점', how='inner'))
