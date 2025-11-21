# nn_param_sweep.py
"""
Neural Network(MLPClassifier) 하이퍼파라미터 민감도 실험용 스크립트

- 공유용 tuning.py 건드리지 않고
- NN 하나만, 파라미터를 한 번에 하나씩 바꿔가며 CV F1을 비교
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# 네 프로젝트 구조에 맞게 import (tuning.py에서 쓰던 그대로)
from common import preprocessing as prep_mod


DATA_PATH = "default of credit card clients.xls"
TARGET_COL = "default payment next month"
SEED = 42


def load_data_and_preprocess():
    """엑셀 로드 + clean_data + 전처리 파이프라인(preprocessor)까지 준비."""
    # 엑셀 로드 (tuning.py랑 동일하게 header=1)
    df = pd.read_excel(DATA_PATH, header=1)

    if TARGET_COL not in df.columns:
        raise ValueError(f"target '{TARGET_COL}' not in columns")

    # target을 살려 둔 상태로 clean_data 적용
    df[TARGET_COL] = df[TARGET_COL]
    df = prep_mod.clean_data(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # ColumnTransformer 기반 전처리
    preprocessor = prep_mod.make_preprocessor(X)

    return X, y, preprocessor


def eval_mlp(params, X, y, preprocessor, cv_splits=3, seed=SEED):
    """
    주어진 MLP 하이퍼파라미터(params)로
    Pipeline(prep -> MLP) 구성해서 F1 CV 점수 평가.
    """
    clf = MLPClassifier(
        random_state=seed,
        early_stopping=True,
        max_iter=300,
        solver="adam",
        **params,  # hidden_layer_sizes, activation, alpha, learning_rate_init, batch_size 등
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf),
    ])

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, scoring="f1", cv=cv, n_jobs=-1)

    return scores.mean(), scores.std()


def main():
    print(">>> 데이터 로드 + 전처리 준비 중...")
    X, y, preprocessor = load_data_and_preprocess()
    print(f"X shape = {X.shape}, y shape = {y.shape}")

    # -------------------------------------------------
    # 1) 기준(base) 하이퍼파라미터 설정
    #    -> 예전에 RandomizedSearch로 괜찮게 나왔던 값 사용
    # -------------------------------------------------
    base_params = {
        "hidden_layer_sizes": (256, 128, 64),
        "activation": "relu",
        "alpha": 1e-5,
        "learning_rate_init": 3.1622776601683794e-4,  # 0.0003162
        "batch_size": 256,
    }

    print("\n[Base 설정]")
    for k, v in base_params.items():
        print(f"  {k}: {v}")

    # -------------------------------------------------
    # 2) 각 파라미터별로 sweep할 값 정의
    #    -> 한 번에 한 파라미터만 바꿔보기
    # -------------------------------------------------
    sweeps = {
        "hidden_layer_sizes": [
            (64,),
            (128,),
            (256,),
            (128, 64),
            (256, 128),
            (256, 128, 64),
        ],
        "activation": [
            "relu",
            "tanh",
        ],
        "alpha": [
            1e-5,
            3.16e-5,
            1e-4,
            3.16e-4,
            1e-3,
        ],
        "learning_rate_init": [
            1e-4,
            3.16e-4,
            1e-3,
            3.16e-3,
            1e-2,
        ],
        "batch_size": [
            64,
            128,
            256,
        ],
    }

    # -------------------------------------------------
    # 3) 파라미터 하나씩 바꿔가며 실험
    # -------------------------------------------------
    results = {}

    for param_name, values in sweeps.items():
        print("\n" + "=" * 60)
        print(f"=== Sweep: {param_name} ===")
        print("=" * 60)

        param_results = []

        for val in values:
            params = base_params.copy()
            params[param_name] = val  # 이 파라미터만 바꾼다

            mean_f1, std_f1 = eval_mlp(params, X, y, preprocessor)

            param_results.append((val, mean_f1, std_f1))
            print(f"{param_name}={val} -> F1 = {mean_f1:.4f} ± {std_f1:.4f}")

        results[param_name] = param_results

    # -------------------------------------------------
    # 4) 원하면 CSV로 저장해서 엑셀/노션에 붙일 수도 있음
    # -------------------------------------------------
    import csv
    from pathlib import Path

    out_dir = Path("nn_sensitivity_results")
    out_dir.mkdir(exist_ok=True)

    for param_name, param_results in results.items():
        csv_path = out_dir / f"{param_name}_sweep.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([param_name, "mean_f1", "std_f1"])
            for val, mean_f1, std_f1 in param_results:
                writer.writerow([repr(val), mean_f1, std_f1])

        print(f"[저장] {param_name} sweep 결과 -> {csv_path}")


if __name__ == "__main__":
    main()
