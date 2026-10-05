import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'D2Coding'
titanic = sns.load_dataset('titanic')
table = titanic.pivot_table(index='class',
                            columns='sex',
                            values='survived')
sns.heatmap(table, annot=True, fmt='.2f', cmap='Reds')
plt.title('객실 등급·성별 생존율')
plt.show()
