# CreditCard-Risk-Scoring
2025 홍익대학교 기계학습기초 팀 프로젝트

📌 Default Credit Card Clients — 신용카드 연체 예측 모델 비교 연구

## 1. 프로젝트 개요

본 프로젝트는 신용카드 고객들의 연체 여부(default)를 예측하는 머신러닝 모델을 비교·분석하는 것을 목표로 진행되었습니다.

사용한 데이터는 Kaggle UCI의 Default of Credit Card Clients Dataset이며,
전체 약 22만 건 중 연체자 비율이 약 2:8로 구성된 대표적인 불균형 데이터셋입니다.

이러한 불균형 문제 때문에 단순 Accuracy가 아닌,
연체자를 놓치지 않는 ‘Recall’과 잘못된 경보를 줄이는 ‘Precision’을 균형 있게 반영하는
📌 F1-score를 최우선 평가 지표로 채택했습니다.

또한 불균형 문제 해결을 위해 다음 두 접근을 실험적으로 비교했습니다:

- Sampling 기반 데이터 증강 (SMOTE, Cluster Centroids)

- Class Weight 기반의 모델 가중치 조정

이 두 전략이 서로 다른 모델에서 어떤 영향을 주는지 정량적으로 비교하는 것이 핵심 목표입니다.


## 2. 실험 파이프라인 설계
🔧 기술적 핵심 — 공정성(Consistency)

4개의 모델(Logistic Regression, Random Forest, XGBoost/LightGBM, Neural Network)을
동일한 조건에서 비교하기 위해 자동화된 파이프라인을 구축했습니다.

파이프라인 구성 요소

1) 통합 전처리 파이프라인

2) Config 기반 실험 환경(config.yaml)

3) 공통 RandomizedSearchCV 코드

4) 동일한 train/test split & 5-Fold Stratified CV 적용

5) Threshold Optimization 공통 적용

이러한 구조를 구축하여 각 모델이 완전히 동일한 환경에서 평가되도록 보장했습니다.



## 3. 폴더링
```
ml-project/
├─ data/                       # 원본/가공 데이터
├─ common/
│  ├─ preprocessing.py         # 공통 ColumnTransformer/SMOTE 스위치 등
│  └─ evaluation.py            # 공통 CV, 메트릭(ROC-AUC, PR-AUC, 리포트)
│  
├─ models/
│  ├─ logistic.py              # build_model() 구현
│  ├─ random_forest.py         # build_model() 구현
│  ├─ xgboost_lightgbm.py      # build_model() 구현(둘 다 리턴 가능)
│  └─ mlp.py                   # build_model() 구현
├─ run_benchmark.py            # 모델들 호출 후 공통 평가
└─ config.yaml                 # 공통 하이퍼파라미터/실험 설정(선택)
```


## 4. 데이터 전처리
변수 유형별 처리

변수 유형	처리 방식
- 명목형 변수:	One-Hot Encoding
- 순서형 변수:	Ordinal Encoding(순서를 유지)
- 수치형 변수:	Imputation, IQR Clipping, StandardScaler

추가 옵션
- PCA(Optional)
- Sampling(Optional): SMOTE / Cluster Centroids

모든 실험 옵션은 config.yaml에서 On/Off 가능하도록 설계해 모델별 실험 유연성을 확보했습니다.



## 5. 검증 전략

약 30,000건 규모의 데이터를 고려하여, 데이터 분포를 유지하면서 신뢰도를 확보하기 위해 📌 Stratified 5-Fold Cross Validation을 적용했습니다.



## 6. 5단계 실험 프로세스


1) Baseline 측정 (튜닝 없음)

2) RandomizedSearchCV 기반 1차 튜닝

3) Grid Search + Fine-tuning

4) SMOTE 적용 성능 측정

5) Cluster Centroids 적용 성능 측정

튜닝 및 최종 성능 평가는 모든 모델에 대해 최적 Threshold 적용 F1-score로 비교했습니다.

## 7. 모델별 실험 결과 및 주요 인사이트

◆ 7-1. Logistic Regression

| 단계                 | F1-score   |
| ------------------ | ---------- |
| Baseline           | 0.4804     |
| RandomizedSearchCV | **0.5320** |
| Fine-Tuning        | 0.5262     |
| SMOTE              | 0.5296     |
| Cluster Centroids  | ↓ 0.4575   |


✔ 핵심 인사이트

- 선형 모델은 결정 경계가 단순해, 소수 클래스 증강(SMOTE)을 했을 때 유일하게 성능이 상승함.

- Cluster Centroids는 정상 고객 정보가 과도하게 손실되어 성능 급락함.

- Logistic Regression에서는 Threshold 최적화 + 미세 조정(C, class_weight) 조합이 가장 효과적임.


◆ 7-2. Random Forest

| 단계                 | F1-score   |
| ------------------ | ---------- |
| Baseline           | 0.5198     |
| RandomizedSearchCV | **0.5604** |
| Fine-Tuning        | 0.5421     |
| SMOTE              | 0.4915     |
| Cluster Centroids  | ↓ 0.4255   |


✔ 핵심 인사이트

- Random Forest는 데이터 증강보다 class_weight 조정만으로도 충분히 불균형을 해결함.

- SMOTE는 오히려 precision을 크게 떨어뜨리는 노이즈로 작용.

- PR-AUC 기준 RandomizedSearch → Threshold Optimization 조합이 가장 효과적.


◆ 7-3. Boosting Models (XGBoost & LightGBM)

| 단계                 | F1-score   |
| ------------------ | ---------- |
| Baseline           | 0.5418     |
| RandomizedSearchCV | **0.5503** |
| Fine-Tuning        | 0.5399     |
| SMOTE              | 0.5151     |
| Cluster Centroids  | ↓ 0.4142   |

✔ 핵심 발견: 트리 깊이(depth)가 불균형 학습에 결정적인 영향

- depth가 깊으면 → 다수 클래스 학습에만 치우침.
- depth를 3으로 얕게 + scale_pos_weight 강화 → 소수 클래스(연체자) 패턴 학습에 집중하며 성능 최대화.

✔ Boosting의 특성

- 오답(Residual)에 집중해 가중치를 조정하는 구조. 즉, 희소한 연체자 패턴 학습에 매우 유리.
- 성능은 Random Forest와 매우 비슷한 수준으로 수렴했으며, 특히 Recall 성능은 RF보다 높았음.


◆ 7-4. Neural Network (MLP)

| 단계                 | F1-score   |
| ------------------ | ---------- |
| Baseline           | 0.5356     |
| RandomizedSearchCV | **0.5473** |
| Fine-Tuning        | 0.5453     |
| SMOTE              | 0.4381     |
| Cluster Centroids  | ↓ 0.4156   |


✔ 핵심 인사이트

- Neural Network는 sampling보다 원본 데이터 보존 + 튜닝 조합에서 성능 극대화.
- SMOTE는 Recall은 올리지만 Precision을 0.29 수준으로 만들며 F1 급락.
- 모델 자체의 표현력이 강해 class_weight 없이도 원본 패턴 학습에 유리함.



## 8. 모델 간 성능 종합 비교 & 관찰된 패턴
🔎 1) Logistic Regression

- 선형성의 한계로 가장 낮은 성능.
- 데이터의 복잡한 패턴을 충분히 설명하지 못함.


🔎 2) Random Forest vs Boosting

Random Forest
- 다수결(Bagging) 기반 → 보수적인 예측.
- Precision이 매우 높아 안정적.

Boosting
- 오답에 집중해 가중치 조정 → 희소 패턴을 적극 학습.
- Recall이 압도적으로 높음.

- 점수만 보면 RF가 소폭 우위지만, 연체자를 놓치는 비용(Recall)이 크다면 Boosting이 더 적절할 수 있음


🔎 3) Boosting & Neural Network의 성능 수렴

구조는 다르지만 두 모델 모두
f1, recall, pr-auc, roc-auc 모두 비슷한 수준으로 수렴함.

👉 이는 데이터가 허용하는 성능 상한선에 도달했을 가능성을 시사.



## 9. 최종 결론

📌 인위적 데이터 증강(SMOTE, CC)은
→ Logistic Regression을 제외하면 대부분의 모델에서 성능을 저하시켰다.


📌 Random Forest, Boosting, Neural Network 모두
→ Class Weight 또는 Hyperparameter 튜닝 기반 접근이 압도적으로 효과적


📌 모델 선택은 “성능이 제일 높은 모델”이 아니라
→ 클라이언트의 목표(Precision vs Recall 중 무엇을 우선하느냐)에 따라 달라진다.


🔥 핵심 결론 및 토론

-불균형 데이터에서 무작정 데이터를 늘리는 것보다 모델이 어떤 부분에 민감하게 반응해야 하는지 조절하는 전략이 더 중요

-동일한 점수라도 해석과 목표 설정이 더 중요



## 10. 기여
김서현 — Boosting 모델 실험, 공통 파이프라인 개발

박병선 — Logistic Regression 실험 및 튜닝 개발

박채아 — Neural Network 실험 및 데이터 전처리

김채원 — Random Forest 실험 및 데이터 전처리



## 11. 기타
✏️ Commit 컨벤션

- **feat**: 새로운 기능 추가
- **fix**: 버그 수정
- **docs**: 문서 변경 (예: README 수정)
- **style**: 코드 포맷팅, 세미콜론 누락 등 비즈니스 로직에 영향을 주지 않는 변경
- **refactor**: 코드 리팩토링
- **test**: 테스트 추가 또는 기존 테스트 수정
- 🔧: 빌드 프로세스 또는 보조 도구와 관련된 변경 (예: 패키지 매니저 설정)

---
 🔖 브랜치 컨벤션
* `main` - main 브랜치
* `feat/xx` - 모델 단위로 독립적인 개발 환경을 위해 작성
* `refac/xx` - 개발된 기능을 리팩토링 하기 위해 작성
* `chore/xx` - 빌드 작업, 패키지 매니저 설정 등




