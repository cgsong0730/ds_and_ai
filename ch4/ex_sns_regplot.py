import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'D2Coding'

iris = sns.load_dataset('iris')
sns.regplot(data=iris, x='petal_length',
            y='petal_width')
plt.title('꽃잎 길이와 너비의 관계')
plt.show()
