import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'D2Coding'
plt.rcParams['axes.unicode_minus'] = False
iris = sns.load_dataset('iris')
corr = iris.drop(columns='species').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('붓꽃 특성 간 상관관계')
plt.show()
