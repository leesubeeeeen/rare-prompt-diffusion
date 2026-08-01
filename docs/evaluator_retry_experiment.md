# Evaluator-guided retry experiment protocol

## Research question

Does evaluator-guided regeneration improve rare-prompt satisfaction over a
single diffusion sample, within a fixed maximum generation budget?

## Required comparisons

For each prompt and base seed, retain the first attempt as the single-sample
baseline. Compare it with the image chosen by the evaluator-guided loop using:

- final total score and each requirement score;
- human preference and human requirement-satisfaction labels;
- acceptance rate; and
- mean attempts used (the compute cost).

The system always records the best candidate if no image passes. This outcome
must be reported separately from accepted outputs.

## Score calibration

The defaults in the sample config are starting values, not universal quality
thresholds. Generate a small held-out calibration set, inspect its images, and
choose a requirement minimum that separates visibly missing requirements from
successful images. Keep that threshold fixed for all test prompts.

## Avoiding evaluator-only claims

CLIP is used here because it is available, reproducible, and supplies a
requirement-level signal. It is not a reliable detector of every anatomical or
spatial error. Report a blinded human evaluation alongside the automatic
scores, especially for relation binding (for example, whether the frog is
actually driving rather than merely appearing near a car).
