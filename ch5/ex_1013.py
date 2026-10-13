import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
df = pd.read_csv('dataset/precipitation.csv')

# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())

bars = sns.barplot(x='month', y='precipitation(mm)', data=df)
plt.title('월별 강수량')
plt.bar_label(bars.containers[0])
plt.show()
