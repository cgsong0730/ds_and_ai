import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'D2Coding'

store = ['강남', '부산', '수원']
sales = [195000, 252000, 250500]
bars = plt.bar(store, sales, color='skyblue')
plt.bar_label(bars)
plt.title('지점별 매출 합계')
plt.show()
