import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'D2Coding'

url = 'https://api.open-meteo.com/v1/forecast'
params = {
    'latitude': 37.57, 
    'longitude': 126.89,
    'hourly': 'temperature_2m',
    'timezone': 'Asia/Seoul',
    'forecast_days': 1
}

res = requests.get(url, params=params)
print(res.status_code) # 200이면 성공
data = res.json() # JSON → 딕셔너리

print(data['hourly_units'])

hourly = pd.DataFrame({
    'time': data['hourly']['time'],
    'temperature_2m': data['hourly']['temperature_2m']
})
print(hourly)

sns.lineplot(data=hourly, x='time', y='temperature_2m', marker='o')
plt.title('서울 시간별 기온 예보')
plt.xlabel('시간 (서울 기준)')
plt.ylabel(f"기온 ({data['hourly_units']['temperature_2m']})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
