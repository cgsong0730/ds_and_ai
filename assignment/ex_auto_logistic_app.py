import math
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "auto_mpg_model.joblib"


def read_number(prompt, minimum, maximum, allowed_values=None):
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("숫자를 입력해 주세요.")
            continue

        if not math.isfinite(value) or not minimum <= value <= maximum:
            print(f"{minimum}부터 {maximum}까지의 숫자를 입력해 주세요.")
        elif allowed_values is not None and value not in allowed_values:
            print(f"다음 값 중 하나를 입력해 주세요: {', '.join(map(str, allowed_values))}")
        else:
            return value


def main():
    if not MODEL_PATH.is_file():
        raise SystemExit(
            f"모델 파일이 없습니다: {MODEL_PATH}\n"
            "먼저 ex_auto_logistic_model.py를 실행하여 모델을 생성해 주세요."
        )

    model = joblib.load(MODEL_PATH)
    print("<연비 계산기>")
    print("자동차 정보를 입력해 주세요. 입력 범위는 결측치를 제외한 데이터셋 기준입니다.")
    print("학습 데이터는 1970~1982년 자동차이므로 다른 시대의 차량에는 적합하지 않을 수 있습니다.")

    car = {
        "cylinders": read_number(
            "실린더 수 (개, 3/4/5/6/8): ", 3, 8, (3, 4, 5, 6, 8)
        ),
        "displacement": read_number(
            "엔진 배기량 (세제곱인치, 68~455): ", 68, 455
        ),
        "horsepower": read_number("엔진 출력 (마력 hp, 46~230): ", 46, 230),
        "weight": read_number("차량 무게 (파운드 lb, 1613~5140): ", 1613, 5140),
        "acceleration": read_number(
            "정지 상태에서 시속 60마일까지 가속 시간 (초, 8~24.8): ", 8, 24.8
        ),
        "model_year": read_number(
            "모델 연식 (연도의 뒤 두 자리, 1970~1982년: 70~82): ",
            70, 82, tuple(range(70, 83)),
        ),
        "origin": read_number(
            "제조 지역 코드 (단위 없음, 1=미국, 2=유럽, 3=일본): ",
            1, 3, (1, 2, 3),
        ),
    }

    new_car = pd.DataFrame([car])
    mpg = model.predict(new_car)[0]
    print(f"예상 연비: {mpg:.2f} mpg (약 {mpg * 0.425144:.2f} km/L)")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 취소되었습니다.")
