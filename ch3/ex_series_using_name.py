import pandas as pd

s = pd.Series([1, 2, 3, 4])
s.name = "number"
s.index.name = "id"
print(s)

"""
id                                                                                                          
0    1
1    2
2    3
3    4
Name: number, dtype: int64
"""
