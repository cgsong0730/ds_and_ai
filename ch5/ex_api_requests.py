import requests

url = 'https://api.open-meteo.com/v1/forecast'
params = {'latitude': 37.57, 'longitude': 126.98,
          'hourly': 'temperature_2m',
          'timezone': 'Asia/Seoul', 'forecast_days': 1}
res = requests.get(url, params=params)
print(res.status_code)                       # 200이면 성공
data = res.json()                            # JSON → 딕셔너리
print(data['hourly_units'])
print(data['hourly']['time'][:3])
print(data['hourly']['temperature_2m'][:3])
