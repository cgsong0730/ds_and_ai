import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 10000))  # 의미 없는 피쳐 10,000개
y = rng.integers(0, 2, 100)        # 무작위 정답 (0 또는 1)

# 잘못된 방법: 전체 데이터로 피쳐를 먼저 고른 뒤 교차 검증
X_sel = SelectKBest(k=20).fit_transform(X, y)
print(round(cross_val_score(LogisticRegression(), X_sel, y, cv=5).mean(), 2))
# 0.89 → 검증 데이터의 정답을 엿보면서 발생한 데이터 누수

# 올바른 방법: 피쳐 선택을 파이프라인에 넣어 조각마다 다시 수행
pipe = make_pipeline(SelectKBest(k=20), LogisticRegression())
print(round(cross_val_score(pipe, X, y, cv=5).mean(), 2))
# 0.47 → 데이터 조각마다 전처리를 새로 학습
