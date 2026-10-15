import io, requests
import pandas as pd
from bs4 import BeautifulSoup
url = 'https://books.toscrape.com/catalogue/its-only-the-himalayas_981/index.html'
res = requests.get(url)
res.encoding = 'utf-8'
soup = BeautifulSoup(res.text, 'html.parser')
info = {}
for tr in soup.select('table.table-striped tr'): # 표의 각 행
    info[tr.select_one('th').text] = tr.select_one('td').text
print(pd.Series(info))
tables = pd.read_html(io.StringIO(res.text)) # 표 전체를 한 번에
print(len(tables), tables[0].shape)
