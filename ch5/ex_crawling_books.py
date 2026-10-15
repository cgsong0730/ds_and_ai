import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://books.toscrape.com/'
res = requests.get(url)
res.encoding = 'utf-8'           # 특수문자(£) 깨짐 방지
soup = BeautifulSoup(res.text, 'html.parser')

rows = []
for book in soup.select('article.product_pod'):
    title = book.select_one('h3 a')['title']
    price = book.select_one('p.price_color').text
    rows.append({'title': title, 'price': price})
df = pd.DataFrame(rows)
df.to_csv('books.csv', index=False)
print(df.head(3))
