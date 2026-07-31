# Rare Prompt Robustness for Text-to-Image Diffusion

서로 어울리지 않는 수식어와 명사의 희귀 조합(예: `the driving frog`, `an octopus with flowing hair`)을 text-to-image diffusion model이 정확하게 표현하도록 만드는 방법을 연구하는 저장소입니다.

## 저장소 구조

```text
methods/
  linear_distribution/   # 종료된 선형 분포 보간 방법
  relation_binding/      # 다음 방법을 위한 작업 공간
configs/
  linear_distribution/   # 종료된 방법의 재현 설정
experiments/
  failed_attempts/       # 실패 실험의 이미지, 지표, 후보 기록
  current/               # 다음 실험의 산출물 위치
docs/
  experiment_log.md      # 가설, 결과, 실패 원인 기록
```

## 현재 상태

`linear_distribution` 방법은 종료된 baseline입니다. 구성요소별 prompt embedding 분포를 추정하고 그 중심 부근의 conditioning을 선택했지만, 명사와 수식어 사이의 관계를 보존하지 못해 두 rare prompt 모두 원본 prompt baseline보다 낮은 CLIP alignment를 기록했습니다.

새로운 방법은 아직 정하지 않았습니다. 다음 실험을 기획한 뒤 `methods/<method_name>/`, `configs/<method_name>/`, `experiments/current/` 아래에 구현과 설정, 결과를 추가합니다.

## 실패 baseline 재현

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python methods/linear_distribution/run_experiment.py \
  --config configs/linear_distribution/rare_driving_frog.json \
  --output-dir experiments/failed_attempts/linear_distribution
```

실패 baseline의 상세 결과와 해석은 [`docs/experiment_log.md`](docs/experiment_log.md)에 기록되어 있습니다.
