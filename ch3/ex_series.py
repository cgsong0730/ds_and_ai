# import pandas as pd

# s = pd.Series([1, 2, 3, 4])
# print(s)

# from pandas import Series
# list_data = [1, 2, 3, 4]
# list_name = ["a", "b", "c", "d"]
# example_obj = Series(data=list_data, index=list_name)
# print(example_obj)

import pandas as pd

s = pd.Series([1, 2, 3, 4])
s.name = "number"
s.index.name = "id"
print(s)
