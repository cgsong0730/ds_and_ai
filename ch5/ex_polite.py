import time
import requests
from urllib import robotparser

UA = 'edu-crawler/1.0 (contact: student@example.com)' # 누가 수집하는지
rp = robotparser.RobotFileParser('https://books.toscrape.com/robots.txt')
rp.read()

def polite_get(url, delay=1.0):
    if not rp.can_fetch(UA, url): # ① robots.txt 확인
        print('수집 금지 경로:', url)
        return None
    time.sleep(delay)  # ② 요청 간격 두기
    return requests.get(url, headers={'User-Agent': UA}, timeout=10) # ③ 신원 표시·시간 제한

res = polite_get('https://books.toscrape.com/catalogue/page-2.html')
print(res.status_code) # 200
