import pandas as pd

# 웹에 있는 CSV 파일의 URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# URL로부터 바로 데이터프레임 생성
df = pd.read_csv(url)

# 데이터프레임 확인
print(df.head())        # 상위 5개 행 출력
print(df.shape)         # (행 개수, 열 개수)
print(df.info())        # 컬럼별 데이터 타입, 결측치 확인
print(df.describe())    # 기술통계 요약 (평균, 표준편차 등)