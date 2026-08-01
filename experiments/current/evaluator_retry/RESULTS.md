# Evaluator-guided retry results (2026-08-01)

## Environment and parameters

- Generator: `runwayml/stable-diffusion-v1-5`
- Evaluator: `openai/clip-vit-base-patch32`
- GPU: NVIDIA A100 80GB PCIe; driver 580.173.02
- PyTorch: 2.11.0a0+NVIDIA 26.03; CUDA build 13.2; CUDA available
- Generation: 512x512, 30 inference steps, guidance 7.5, four base seeds,
  at most four attempts, retry seed stride 1009
- Score weights: alignment 0.35, requirements 0.50, usability 0.15
- Calibrated operational cutoffs: total 0.68, subject 0.64, relation/attribute 0.68

The first attempt for each base seed is the single-sample baseline. The final
image is the highest-total-score candidate observed before acceptance or budget
exhaustion. “Accepted” below is only the configured automatic decision.

## Threshold calibration

The initial frog run used total/requirement thresholds 0.76/0.73. Its 16
candidates occupied only 0.6683--0.6881 total, 0.6389--0.6593 frog, and
0.6754--0.6907 driving-relation score, so all four samples exhausted the retry
budget and none passed. This included images with an unmistakable frog, showing
that the starting requirement cutoff was too high for this evaluator's score
scale.

The calibrated run lowered the total cutoff to 0.68, the subject cutoff to 0.64,
and the relation/attribute cutoff to 0.68. The same values were kept for the
second prompt rather than tuning on its outputs. The octopus result (0%
acceptance despite visually obvious octopuses) shows that even these fixed
absolute cutoffs do not transfer reliably across prompt concepts.

## Baseline versus selected

| Prompt | Mean baseline total | Mean selected total | Delta | Mean attempts | Mean retries | Automatic acceptance |
|---|---:|---:|---:|---:|---:|---:|
| frog driving a car (initial thresholds) | 0.6798 | 0.6868 | +0.0070 | 4.00 | 3.00 | 0/4 (0%) |
| frog driving a car (calibrated) | 0.6798 | 0.6869 | +0.0071 | 2.25 | 1.25 | 4/4 (100%) |
| octopus with flowing hair | 0.6625 | 0.6711 | +0.0086 | 4.00 | 3.00 | 0/4 (0%) |

Calibrated frog seed-level totals were 0.6782→0.6873 (seed 13),
0.6769→0.6861 (29), 0.6774→0.6875 (47), and 0.6868→0.6868 (71).
Octopus totals were 0.6507→0.6599 (17), 0.6616→0.6802 (31),
0.6648→0.6713 (53), and 0.6728→0.6730 (79).

Across the two calibrated prompt runs, mean attempts were 3.125, mean retries
were 2.125, and automatic acceptance was 4/8 (50%). The mean selected-score
gain was +0.00784. This is a small evaluator-score improvement, not evidence of
equivalent human-perceived improvement.

## Visual review

All attempt images and all selected images were inspected in seed/attempt order.
Labels are intentionally strict: a nearby steering wheel is not “driving,” and
tentacles or hair on a human are not “an octopus with hair.”

| Prompt | Baseline human result | Selected human result | Automatic/human agreement |
|---|---|---|---|
| frog driving | 4/4 show a frog; 0/4 clearly show the frog holding the wheel and actively driving | 4/4 show a frog; 0/4 satisfy the driving relation | Disagrees: automatic accepted 4/4 after calibration |
| octopus with flowing hair | 0/4 satisfy both subject and attached-hair attribute | 0/4 satisfy both; seeds 17 and 53 are octopuses without hair, while 31 and 79 are human/tentacle or ambiguous hybrids | Agrees on rejection count, but score ranking does not reliably prefer correct binding |

For frog seed 47, the automatic selection placed a frog on/in front of the
steering wheel rather than actively driving. For octopus seed 31, the
highest-scoring selected image is a woman with tentacles and long hair rather
than an octopus with hair. These are concrete CLIP relation/attribute-binding
false positives. Image sharpness and prompt co-occurrence can raise the total
without satisfying the requested composition.

## Conclusion and limitations

Retry improved the mean automatic score for every prompt group, but this small
experiment does not demonstrate improved rare-prompt satisfaction under human
review. Stable Diffusion v1.5 often produced object co-occurrence or hybrids
instead of the requested relation. CLIP similarity was sensitive to the words'
visual concepts but weak at subject-relation and subject-attribute binding.
Four seeds per prompt are also too few for a general effectiveness claim, and
retrying with a new seed confounds prompt revision with ordinary resampling.

Useful next steps are: add a no-revision random-resampling control with the same
generation budget; evaluate many more held-out seeds and prompts; use blinded,
independent human labels; replace absolute CLIP cutoffs with per-requirement
calibration or a relation-aware VLM/object detector; require evaluator agreement
or uncertainty margins; and test structured prompt edits, attention guidance,
or stronger generators. Report compute-normalized success as well as best-of-N
scores.

## Artifacts

Each experiment directory contains `attempts/`, `selected/`,
`attempt_metrics.csv`, `selected_metrics.csv`, and `summary.json`. The
uncalibrated frog directory is retained unchanged; calibrated reruns use a
different experiment name. No prior result was deleted or overwritten.
