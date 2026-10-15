import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
df = pd.read_csv('dataset/precipitation.csv')

bars = sns.barplot(x='month', y='precipitation(mm)', data=df)
plt.xticks(fontsize=8, fontstyle='italic')
plt.title('월별 강수량')
plt.show()

# URL: https://data.kma.go.kr/stcs/grnd/grndRnList.do?pgmNo=69