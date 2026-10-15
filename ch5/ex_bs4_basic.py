from bs4 import BeautifulSoup

html = """
<ul id="news">
  <li class="item"><a href="/n/1">공공데이터 개방 확대</a></li>
  <li class="item"><a href="/n/2">AI 학습데이터 가이드 발표</a></li>
</ul>
"""
soup = BeautifulSoup(html, 'html.parser')
for a in soup.select('#news .item a'):
    print(a.text, a['href'])
