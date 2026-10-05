import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'D2Coding'

iris = sns.load_dataset('iris')
sns.scatterplot(data=iris, x='petal_length',
                y='petal_width', hue='species',
                style='species')
plt.title('품종별 꽃잎 길이와 너비')
plt.show()
