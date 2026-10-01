import pandas as pd

ages = pd.Series([15, 23, 37, 45, 62, 71])
bins = [0, 19, 39, 59, 100]
labels = ['청소년', '청년', '중년', '노년']
print(pd.cut(ages, bins=bins, labels=labels))
