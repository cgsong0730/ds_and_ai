import csv
import os

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'precipitation.csv')


def read_rows(encoding):
    with open(path, encoding=encoding, newline='') as f:
        return list(csv.reader(f))


try:
    rows = read_rows('utf-8-sig')
except UnicodeDecodeError:
    rows = read_rows('cp949')

header, data = rows[0], rows[1:]
months = [r[0] for r in data]
rain = [float(r[2]) for r in data]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(months, rain, color='skyblue')
ax.bar_label(bars)
ax.plot(months, rain, color='royalblue', marker='o')
ax.set_title('월별 강수량')
ax.set_xlabel(header[0])
ax.set_ylabel(header[2])
plt.tight_layout()
plt.show()
