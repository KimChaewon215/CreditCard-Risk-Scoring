# single_parameter_sweep.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline  # Pipeline 임포트 추가

from common import preprocessing as prep_mod


# --------------------------
# 1) 데이터 로드 + clean_data
# --------------------------
def load_data():
    """데이터를 로드하고 clean_data를 적용한 후 X, y를 반환합니다."""
    # 파일 경로를 환경에 맞게 조정해야 할 수도 있습니다.
    df = pd.read_excel("default of credit card clients.xls", header=1)
    target = "default payment next month"

    df[target] = df[target]  # 타겟 컬럼 임시 복사
    df = prep_mod.clean_data(df)  # Clean data 적용

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# --------------------------
# 2) CV 평가 함수 (Pipeline 사용으로 수정)
# --------------------------
def evaluate_model(clf_params, X, y):
    """
    3-Fold Stratified K-Fold Cross-Validation을 수행하고 평균 점수를 반환합니다.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    f1_scores = []
    roc_scores = []
    pr_scores = []

    # 기본 모델 설정 (evaluate_model 함수에 파라미터를 dict로 전달받음)
    base_clf = RandomForestClassifier(random_state=42, n_jobs=-1, **clf_params)

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        # 1. 전처리 객체 생성 (ColumnTransformer는 훈련 데이터로만 fit)
        preprocessor = prep_mod.make_preprocessor(X_tr)

        # 2. Pipeline으로 전처리 단계와 모델을 결합
        pipeline = Pipeline(steps=[
            ('prep', preprocessor),
            ('clf', base_clf)
        ])

        # Pipeline 학습 및 예측
        # X_tr (DataFrame)이 Pipeline의 'prep' 단계를 거쳐 'clf' 모델로 전달됩니다.
        pipeline.fit(X_tr, y_tr)
        proba = pipeline.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        # 스코어 저장
        f1_scores.append(f1_score(y_te, pred))
        roc_scores.append(roc_auc_score(y_te, proba))
        pr_scores.append(average_precision_score(y_te, proba))

    return np.mean(f1_scores), np.mean(roc_scores), np.mean(pr_scores)


# --------------------------
# 3) 파라미터 sweep 함수 (수정됨: 파라미터 딕셔너리 전달)
# --------------------------
def sweep_parameter(param_name, param_values):
    """지정된 단일 파라미터 값들에 대해 모델을 평가하고 결과를 시각화합니다."""
    X, y = load_data()

    results = {
        "value": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": []
    }

    # 기본 파라미터 설정 (튜닝하지 않는 파라미터의 고정 값)
    base_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
    }

    for val in param_values:
        print(f"Testing {param_name} = {val}")

        # 기본 파라미터 딕셔너리를 복사하고, 현재 sweep할 파라미터만 업데이트
        current_params = base_params.copy()
        current_params[param_name] = val

        # evaluate_model에 파라미터 딕셔너리를 전달
        f1, roc, pr = evaluate_model(current_params, X, y)

        results["value"].append(val)
        results["f1"].append(f1)
        results["roc_auc"].append(roc)
        results["pr_auc"].append(pr)

    df = pd.DataFrame(results)
    print("\n=== 결과표 ===")
    print(df.round(4))

    # 그래프 출력
    plt.figure(figsize=(8, 5))

    # x축을 문자열로 변환 (max_depth의 None 값 처리)
    x_labels = [str(v) for v in results["value"]]
    x_positions = range(len(x_labels))

    plt.plot(x_positions, results["f1"], marker='o', label="F1")
    plt.plot(x_positions, results["roc_auc"], marker='o', label="ROC-AUC")
    plt.plot(x_positions, results["pr_auc"], marker='o', label="PR-AUC")

    plt.xticks(x_positions, x_labels)  # x축 레이블 설정

    plt.title(f"Parameter Sweep: {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df


# --------------------------
# 4) 실행 예시
# --------------------------
if __name__ == "__main__":
    # n_estimators 실험
    sweep_parameter(
        param_name="n_estimators",
        param_values=[50, 100, 200, 400, 800]
    )