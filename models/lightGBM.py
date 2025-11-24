from lightgbm import LGBMClassifier


def build_model(**kwargs):
    defaults = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,  # XGBoost랑 조건 맞추려고 4로 설정 (LGBM은 원래 -1이 기본)
        num_leaves=20,  # ✨ 중요: LGBM의 핵심! (보통 2^max_depth보다 작게 설정)
        min_child_samples= 10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,  # L2 Regularization
        reg_alpha=0.0,  # L1 Regularization
        random_state=42,
        n_jobs=2,
        verbose=-1

        # LightGBM 전용 추가 옵션 (필요하면 주석 해제)
        # importance_type='gain',  # 변수 중요도 볼 때 gain 기준이 더 정확할 때가 많음
        # boosting_type='gbdt'     # 기본값
    )

    defaults.update(kwargs)  # YAML의 값으로 덮어쓰기

    return LGBMClassifier(**defaults)