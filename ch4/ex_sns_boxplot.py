import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'D2Coding'

titanic = sns.load_dataset('titanic')
sns.boxplot(data=titanic, x='class', y='age')
plt.title('객실 등급별 나이 분포')
plt.show()
