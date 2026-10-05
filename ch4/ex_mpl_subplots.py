import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'D2Coding'
titanic = sns.load_dataset('titanic')
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

axes[0].hist(titanic['age'].dropna(), bins=20)
axes[0].set_title('나이 분포')
axes[1].bar(['남성', '여성'], [577, 314])
axes[1].set_title('성별 승객 수')

plt.tight_layout()
plt.show()
