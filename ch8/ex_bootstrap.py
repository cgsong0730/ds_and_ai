import numpy as np

rng = np.random.default_rng(42)
idx = rng.choice(712, size=712, replace=True)  # 712번 복원 추출
print("뽑힌 승객 수:", len(np.unique(idx)))
print("OOB 승객 수:", 712 - len(np.unique(idx)))
# 뽑힌 승객 수: 446
# OOB 승객 수: 266
