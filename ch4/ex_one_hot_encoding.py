import pandas as pd

df = pd.DataFrame({'색상': ['빨강', '초록', '파랑', '초록']})
print(pd.get_dummies(df['색상'], dtype=int))

"""
   빨강  초록  파랑
0     1     0     0
1     0     1     0
2     0     0     1
3     0     1     0
"""