# Experiment log

## Linear valid-distribution baseline

### 가설

Rare prompt를 구성요소로 분해하고 각 구성요소의 CLIP text-conditioning 분포를 추정한다. 각 분포 peak의 선형 중심 주변에서 결합 밀도가 높은 conditioning을 선택하면 원본 rare prompt보다 강건한 이미지를 생성할 수 있다고 가정했다.

### 수행한 실험

| Prompt | Baseline CLIP | Linear-valid CLIP | 결과 |
|---|---:|---:|---:|
| `the driving frog` | 29.478 | 21.675 | -7.803 |
| `an octopus with flowing hair` | 32.301 | 26.875 | -5.426 |

초기의 `a bioluminescent medieval submarine library` 실험은 수식어-명사 관계가 충분히 희귀하지 않아 연구 대상 정의와 맞지 않았다. 이 결과도 실험 선택 오류의 기록으로 보존한다.

### 실패 원인

1. 서로 다른 문장의 `[77, hidden]` token embedding을 같은 위치끼리 평균해 문장 구조와 token 의미를 손상했다.
2. 개별 개념 분포 사이의 고밀도 타협점이 유효한 text-conditioning manifold에 있다는 보장이 없다.
3. 개념의 공존만 유도할 뿐 `frog`가 운전하거나 `hair`가 `octopus`에 붙는 명사-수식어 관계를 표현하지 못했다.
4. 후보 선택 목적함수에 원본 rare prompt와의 정렬이나 관계 정확도가 포함되지 않았다.
5. 전체 prompt CLIP score만으로는 객체 존재, 속성 존재, 관계 binding을 분리해 평가할 수 없다.

### 결정

이 방법은 주 연구 방법으로 폐기하고 negative baseline 및 ablation 용도로만 보존한다. 새로운 방법은 별도의 기획 후 구현한다.
