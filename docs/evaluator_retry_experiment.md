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

### Calibration used in the 2026-08-01 run

The initial frog run used a total acceptance threshold of 0.76 and per-requirement
minimums of 0.73. All 16 candidates scored only 0.6683--0.6881 total,
0.6389--0.6593 for the frog requirement, and 0.6754--0.6907 for the driving
relation. Consequently every output exhausted all four attempts and the
acceptance rate was 0%, including images where the frog was plainly visible.

For the follow-up run, the operational thresholds were changed to 0.68 total,
0.64 for the visible subject, and 0.68 for the relation. These cutoffs lie within
the observed score range and allow the loop to react to relative score changes.
They are not calibrated proof of semantic success: visual review found that even
high-scoring candidates often put a frog near or on a steering wheel without
showing it actively driving. Therefore automatic acceptance and human
requirement satisfaction are reported separately. The same 0.64/0.68
requirement thresholds and 0.68 total threshold are fixed for the second prompt
to avoid prompt-by-prompt tuning on test outputs.

## Avoiding evaluator-only claims

CLIP is used here because it is available, reproducible, and supplies a
requirement-level signal. It is not a reliable detector of every anatomical or
spatial error. Report a blinded human evaluation alongside the automatic
scores, especially for relation binding (for example, whether the frog is
actually driving rather than merely appearing near a car).
