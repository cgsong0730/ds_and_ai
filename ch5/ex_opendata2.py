import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터 읽기
df = pd.read_csv('dataset/subway_congestion_level.csv', encoding='cp949')

# 2. 안암역 행만 고르기
anam = df[df['출발역'] == '안암']
print(anam.iloc[:, :5])

# 3. 시간대 열(6번째 열부터)
times = df.columns[5:]

# 4. 행마다 선 그래프 그리기
plt.figure(figsize=(14, 6))
for i in anam.index:
    label = anam.loc[i, '요일구분'] + ' ' + anam.loc[i, '상하구분']
    plt.plot(times, anam.loc[i, times].astype(float), label=label)

plt.title('안암역 시간대별 혼잡도')
plt.xlabel('시간대')
plt.ylabel('혼잡도')
plt.xticks(rotation=90, fontsize=8)
plt.legend()
plt.tight_layout()
plt.show()

# URL: https://www.data.go.kr, 서울교통공사_지하철혼잡도정보