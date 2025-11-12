from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier

def build_model(*, n_estimators=400, max_depth=None,
                min_samples_split=2, min_samples_leaf=1,
                max_features="sqrt", class_weight="balanced",
                random_state=42, n_jobs=-1) -> RandomForestClassifier:
    """
    RandomForest 모델 객체만 반환 (전처리 포함 X)
    run_benchmark.py에서 외부 파이프라인과 결합됨.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )