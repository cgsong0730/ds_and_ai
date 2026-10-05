import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'D2Coding'
iris = sns.load_dataset('iris')
for name, group in iris.groupby('species'):
    plt.scatter(group['petal_length'],
                group['petal_width'], label=name)
plt.xlabel('꽃잎 길이(cm)')
plt.ylabel('꽃잎 너비(cm)')
plt.legend()
plt.show()
