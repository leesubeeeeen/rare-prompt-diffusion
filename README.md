# Rare Prompt Robustness: Linear Valid-Distribution Baseline

희귀한 조합 프롬프트를 구성요소로 분해하고, 각 구성요소의 CLIP text-conditioning 분포를 추정한 뒤 선형 보간 경로의 중심 부근에서 높은 밀도를 갖는 conditioning을 선택해 이미지를 생성하는 실험입니다.

## 구현한 가설

`rare prompt`의 원본 conditioning과, 구성요소 분포의 peak를 선형 결합한 conditioning을 비교합니다. 선형 중심점 `m` 주위의 후보 `z` 중 아래 점수가 높은 후보를 선택합니다.

`score(z) = log p_A(z) + log p_B(z) + ... - center_penalty * ||z - m||²`

각 `p_i`는 해당 구성요소의 프롬프트 embedding에 적합한 shrinkage Gaussian입니다. 후보는 임의의 embedding noise가 아니라, 각 구성요소의 실제 프롬프트 embedding 선형 결합으로 만들므로 conditioning 형태를 유지합니다.

## 빠른 실행

GPU Linux 세션에서 다음을 실행합니다.

```bash
git clone <YOUR_REPOSITORY_URL> rare-prompt-linear
cd rare-prompt-linear
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiment.py --config configs/example_bioluminescent_submarine.json
```

결과는 `outputs/<experiment_name>/`에 저장됩니다.

## GPU 사이트에서의 순서

1. **스토리지 폴더**를 만들고 이 프로젝트를 GitHub에 push한 뒤 URL로 가져오거나, 폴더 전체를 업로드합니다. 결과와 Hugging Face cache를 스토리지에 두면 세션 삭제 후에도 남습니다.
2. **Interactive 세션**을 먼저 만듭니다. Ubuntu + PyTorch/CUDA 이미지를 고르고 GPU 1장, VRAM 16GB 이상이면 권장합니다. 8GB도 SD 1.5, 512px, batch 1로 가능합니다.
3. 터미널에서 위 빠른 실행 명령을 수행합니다. 처음 실행 시 모델 다운로드가 발생합니다.
4. 결과가 확인된 뒤, 같은 명령을 Batch 세션의 command로 옮겨 여러 config를 실행합니다.

Hugging Face 모델 접근 승인이 필요한 경우 `huggingface-cli login`으로 본인 토큰을 로그인하세요. 토큰을 코드나 Git에 저장하지 마세요.

## 실험 산출물

- `baseline/`: 원본 rare prompt 이미지
- `linear_valid/`: 선택된 선형-valid conditioning 이미지
- `candidates.json`: 후보 가중치·분포 점수·선택 결과
- `metrics.csv`: CLIP text-image alignment
- `summary.json`: 선택 후보 및 요약

## 다음 실험

- `configs/`에 rare prompt별 JSON을 추가합니다.
- seed 수를 4 이상으로 늘립니다.
- `linear_valid`와 baseline의 CLIPScore, 사람 평가(구성요소 충실도/자연스러움)를 비교합니다.
- 선형 보간과 SLERP 또는 원본 prompt embedding을 별도 ablation으로 비교합니다.
