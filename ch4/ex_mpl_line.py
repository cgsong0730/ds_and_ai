import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'D2Coding'
month = [1, 2, 3, 4, 5, 6]
gangnam = [120, 135, 150, 145, 170, 190]
suwon = [100, 110, 105, 130, 140, 160]

plt.plot(month, gangnam, marker='o', label='강남')
plt.plot(month, suwon, marker='s', linestyle='--', label='수원')
plt.title('지점별 월 매출(만 원)')
plt.legend()
plt.show()
