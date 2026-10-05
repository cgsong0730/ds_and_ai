import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'D2Coding'

titanic = sns.load_dataset('titanic')
counts = titanic['class'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%.1f%%')
plt.title('객실 등급별 승객 비율')
plt.show()
