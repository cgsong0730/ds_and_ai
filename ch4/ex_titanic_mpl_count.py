import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'D2Coding'

titanic = sns.load_dataset('titanic')
counts = titanic['survived'].value_counts().sort_index()
bars = plt.bar(['사망', '생존'], counts,
               color=['gray', 'tomato'])
plt.bar_label(bars)
plt.title('타이타닉 생존자와 사망자 수')
plt.show()
