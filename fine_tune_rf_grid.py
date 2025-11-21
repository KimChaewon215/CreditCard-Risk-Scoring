# fine_tune_rf_grid.py
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from common import preprocessing as prep_mod


# --------------------------
# 1) 데이터 로드 + clean_data
# --------------------------
def load_data():
    df = pd.read_excel("default of credit card clients.xls", header=1)
    target = "default payment next month"

    df[target] = df[target]
    df = prep_mod.clean_data(df)

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# --------------------------
# 2) CV 평가 함수
# --------------------------
def evaluate_model(params, X, y):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    f1_scores, roc_scores, pr_scores = [], [], []

    base_clf = RandomForestClassifier(
        random_state=42, n_jobs=-1, **params
    )

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        preprocessor = prep_mod.make_preprocessor(X_tr)

        clf = clone(base_clf)

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf", clf)
        ])

        pipeline.fit(X_tr, y_tr)
        proba = pipeline.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        f1_scores.append(f1_score(y_te, pred))
        roc_scores.append(roc_auc_score(y_te, proba))
        pr_scores.append(average_precision_score(y_te, proba))

    return np.mean(f1_scores), np.mean(roc_scores), np.mean(pr_scores)


# --------------------------
# 3) Fine Grid Search (업데이트 버전)
# --------------------------
def run_fine_grid():
    X, y = load_data()

    # 최적 조합 기반 고정 파라미터
    base_params = {
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': "balanced_subsample"
    }

    # 개선된 탐색 범위
    max_depth_list = [10, 12, 15]        # 8 제거
    n_estimators_list = [600, 800, 1000] # 400 제거

    results = []

    for max_d, n_est in itertools.product(max_depth_list, n_estimators_list):
        print(f"Testing max_depth={max_d}, n_estimators={n_est}")

        params = base_params.copy()
        params["max_depth"] = max_d
        params["n_estimators"] = n_est

        f1, roc, pr = evaluate_model(params, X, y)

        results.append({
            "max_depth": max_d,
            "n_estimators": n_est,
            "f1": f1,
            "roc_auc": roc,
            "pr_auc": pr
        })

    df = pd.DataFrame(results)
    print("\n=== Grid Search Results (Updated) ===")
    print(df.sort_values(by="f1", ascending=False).round(4))

    # 저장
    output_dir = './artifacts'
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "rf_fine_tune_results_updated.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path}")

    # Heatmap 생성
    df_pivot = df.pivot_table(
        values='f1',
        index='max_depth',
        columns='n_estimators'
    )

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        df_pivot,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        cbar_kws={'label': 'F1 Score (Mean)'}
    )
    plt.title('F1 Score Heatmap (Updated Grid)')
    img_path = os.path.join(output_dir, "rf_fine_tune_heatmap_updated_f1.png")
    plt.savefig(img_path)
    print(f"Saved heatmap to: {img_path}")
    plt.show()

    return df


if __name__ == "__main__":
    run_fine_grid()
