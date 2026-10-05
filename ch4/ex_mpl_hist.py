import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'D2Coding'

titanic = sns.load_dataset('titanic')
plt.hist(titanic['age'].dropna(), bins=20,
         edgecolor='black')
plt.title('타이타닉 승객 나이 분포')
plt.xlabel('나이')
plt.show()
