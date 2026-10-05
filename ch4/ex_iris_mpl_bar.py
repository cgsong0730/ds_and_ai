import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'D2Coding'

iris = sns.load_dataset('iris')
mean = iris.groupby('species')['petal_length'].mean()
bars = plt.bar(mean.index, mean.values, color='orchid')
plt.bar_label(bars, fmt='%.2f')
plt.title('품종별 평균 꽃잎 길이(cm)')
plt.show()
