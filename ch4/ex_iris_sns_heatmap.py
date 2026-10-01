import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'D2Coding'

iris = sns.load_dataset('iris')
mean = iris.groupby('species').mean()
sns.heatmap(mean, annot=True, cmap='YlGnBu')
plt.title('품종별 특성 평균(cm)')
plt.show()
