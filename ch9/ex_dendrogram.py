import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
names = ['chevrolet impala', 'ford galaxie 500', 'plymouth fury iii', 'pontiac catalina', 'ford maverick', 'plymouth valiant', 'amc hornet', 'chevrolet nova', 'honda civic', 'toyota corolla', 'datsun 510', 'volkswagen rabbit']
features = ["mpg", "horsepower", "weight", "acceleration"]
df = pd.read_csv(url).dropna().drop_duplicates("name")

cars = df.set_index("name").loc[names, features] # 12대
X = StandardScaler().fit_transform(cars) # 단위가 달라 표준화
Z = linkage(X, method="ward") # 병합 과정 계산
dendrogram(Z, labels=cars.index, orientation="right")

plt.show()