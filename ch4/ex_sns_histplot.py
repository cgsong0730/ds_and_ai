import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'D2Coding'

titanic = sns.load_dataset('titanic')
sns.histplot(data=titanic, x='age', hue='survived',
             kde=True)
plt.title('생존 여부별 나이 분포')
plt.show()
