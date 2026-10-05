import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'D2Coding'

titanic = sns.load_dataset('titanic')
sns.countplot(data=titanic, x='class', hue='survived')
plt.title('객실 등급별 생존자·사망자 수')
plt.show()
