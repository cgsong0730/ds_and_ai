import pandas as pd

scores = {"Korean": 90, "English": 85, "Math": 95}
series = pd.Series(scores)
series.index.name = "course"
series.name = "cgs"
print(series)
