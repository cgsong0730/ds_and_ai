from pandas import Series
list_data = [1, 2, 3, 4]
list_name = ["a", "b", "c", "d"]
example_obj = Series(data=list_data, index=list_name)
print(example_obj)
"""
a    1                                                                                                      
b    2                                                                                                      
c    3                                                                                                      
d    4                                                                                                      
dtype: int64
"""
