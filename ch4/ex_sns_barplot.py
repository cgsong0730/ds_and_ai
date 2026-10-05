import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'D2Coding'

titanic = sns.load_dataset('titanic')
sns.barplot(data=titanic, x='class', y='survived',
            hue='sex')
plt.title('객실 등급·성별 생존율')
plt.show()
