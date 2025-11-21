"""
Neural Network(MLPClassifier) 하이퍼파라미터 민감도 실험용 스크립트

- 공유용 tuning.py 건드리지 않고
- NN 하나만, 파라미터를 한 번에 하나씩 바꿔가며
  CV PR-AUC(average precision)을 비교
"""

import pandas as pd
import numpy as np



from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score

from common import preprocessing as prep_mod

import matplotlib.pyplot as plt


DATA_PATH = "default of credit card clients.xls"
TARGET_COL = "default payment next month"
SEED = 42


def load_data_and_preprocess():
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
    Pipeline(prep -> MLP) 구성해서 CV PR-AUC(average precision) 평가.
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

    # scikit-learn 버전 이슈 피하려고, make_scorer 안 쓰고
    # "estimator, X, y" 형태의 커스텀 스코어 함수를 직접 정의
    def pr_auc_scorer(estimator, X_val, y_val):
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X_val)[:, 1]
        elif hasattr(estimator, "decision_function"):
            proba = estimator.decision_function(X_val)
        else:
            proba = estimator.predict(X_val)
        return average_precision_score(y_val, proba)

    scores = cross_val_score(
        pipe,
        X,
        y,
        scoring=pr_auc_scorer,  # callable 그대로 전달
        cv=cv,
        n_jobs=-1,
    )

    return scores.mean(), scores.std()

def plot_sweep_results(results):
    """하이퍼파라미터별 PR-AUC 민감도 플롯 그리기 (Fig1~Fig5)."""

    # -----------------------------
    # Fig 1. hidden_layer_sizes vs PR-AUC
    # -----------------------------
    param_name = "hidden_layer_sizes"
    if param_name in results:
        vals = results[param_name]
        x_labels = [str(v[0]) for v in vals]   # '(64,)', '(128, 64)' 이런 식
        means = [v[1] for v in vals]

        plt.figure(figsize=(8, 4))
        plt.plot(x_labels, means, marker="o")
        plt.title("Fig 1. Hidden layer sizes vs PR-AUC")
        plt.xlabel("hidden_layer_sizes")
        plt.ylabel("PR-AUC")
        plt.ylim(0.53, 0.56)  # 대략 범위 맞춰주면 변화가 더 잘 보임
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Fig 2. alpha (log scale) vs PR-AUC
    # -----------------------------
    param_name = "alpha"
    if param_name in results:
        vals = results[param_name]
        x = [v[0] for v in vals]   # 실제 alpha 값 (float)
        means = [v[1] for v in vals]

        plt.figure(figsize=(8, 4))
        plt.plot(x, means, marker="o")
        plt.xscale("log")
        plt.title("Fig 2. Alpha (L2 regularization) vs PR-AUC")
        plt.xlabel("alpha (log scale)")
        plt.ylabel("PR-AUC")
        plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Fig 3. batch_size vs PR-AUC
    # -----------------------------
    param_name = "batch_size"
    if param_name in results:
        vals = results[param_name]
        x = [v[0] for v in vals]      # 64, 128, 256
        means = [v[1] for v in vals]

        plt.figure(figsize=(6, 4))
        plt.bar([str(v) for v in x], means)
        plt.title("Fig 3. Batch size vs PR-AUC")
        plt.xlabel("batch_size")
        plt.ylabel("PR-AUC")
        plt.ylim(0.54, 0.555)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Fig 4. activation vs PR-AUC
    # -----------------------------
    param_name = "activation"
    if param_name in results:
        vals = results[param_name]
        x_labels = [str(v[0]) for v in vals]   # 'relu', 'tanh'
        means = [v[1] for v in vals]

        plt.figure(figsize=(5, 4))
        plt.bar(x_labels, means)
        plt.title("Fig 4. Activation function vs PR-AUC")
        plt.xlabel("activation")
        plt.ylabel("PR-AUC")
        plt.ylim(0.548, 0.552)  # 거의 비슷해서 범위 좁게 잡아줌
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Fig 5. learning_rate_init vs PR-AUC
    # -----------------------------
    param_name = "learning_rate_init"
    if param_name in results:
        vals = results[param_name]
        x = [v[0] for v in vals]
        means = [v[1] for v in vals]

        plt.figure(figsize=(8, 4))
        plt.plot(x, means, marker="o")
        plt.xscale("log")
        plt.title("Fig 5. Learning rate vs PR-AUC")
        plt.xlabel("learning_rate_init (log scale)")
        plt.ylabel("PR-AUC")
        plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

def main():
    print(">>> 데이터 로드 + 전처리 준비 중...")
    X, y, preprocessor = load_data_and_preprocess()
    print(f"X shape = {X.shape}, y shape = {y.shape}")

    # -------------------------------------------------
    # 1) 기준(base) 하이퍼파라미터 설정
    #    -> RandomizedSearch로 얻은 최적 값 기준
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

            mean_pr, std_pr = eval_mlp(params, X, y, preprocessor)

            param_results.append((val, mean_pr, std_pr))
            print(f"{param_name}={val} -> PR-AUC = {mean_pr:.4f} ± {std_pr:.4f}")

        results[param_name] = param_results

    # -------------------------------------------------
    # 4) CSV로 저장 (엑셀/노션 등에서 보기 좋게)
    # -------------------------------------------------
    import csv
    from pathlib import Path

    out_dir = Path("nn_sensitivity_results")
    out_dir.mkdir(exist_ok=True)

    for param_name, param_results in results.items():
        csv_path = out_dir / f"{param_name}_sweep.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([param_name, "mean_pr_auc", "std_pr_auc"])
            for val, mean_pr, std_pr in param_results:
                writer.writerow([repr(val), mean_pr, std_pr])

        print(f"[저장] {param_name} sweep 결과 -> {csv_path}")

        # 5) 결과 플롯 그리기
    plot_sweep_results(results)


if __name__ == "__main__":
    main()