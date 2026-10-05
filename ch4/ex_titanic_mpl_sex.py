import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'D2Coding'
titanic = sns.load_dataset('titanic')
rate = titanic.groupby('sex')['survived'].mean()
bars = plt.bar(['여성', '남성'], rate,
               color=['tomato', 'steelblue'])
plt.bar_label(bars, fmt='%.2f')
plt.ylim(0, 1)
plt.title('성별 생존율')
plt.show()
