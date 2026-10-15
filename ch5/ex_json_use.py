import json
import pandas as pd
with open('dataset/orders.json', encoding='utf-8') as f:
    orders = json.load(f) # → 리스트·딕셔너리
orders = [o for o in orders if '상품' in o] # 상품이 없는 주문(1002) 제외
for o in orders: # 주문별 금액 계산
    total = sum(p['가격'] * p['수량'] for p in o['상품'])
    print(o['주문번호'], o['고객']['이름'], total, o['포장'])
items = pd.json_normalize(orders, record_path='상품', meta=['주문번호', ['고객', '이름']])
print(items)
