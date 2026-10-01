import pandas as pd

df1 = pd.DataFrame({'지점': ['강남', '수원', '부산'],
                    '매출': [195000, 250500, 252000]})
df2 = pd.DataFrame({'지점': ['강남', '수원', '대전'],
                    '지역': ['서울', '경기', '대전']})
print(pd.concat([df1, df1], axis=0).shape)  

# (6, 2)