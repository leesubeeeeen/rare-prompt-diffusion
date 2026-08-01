# Evaluator-guided retry

This method generates an image, evaluates prompt alignment and explicit rare-prompt requirements, and retries candidates that fail the configured quality bar. Failed requirements become extra emphasis in the next prompt.

```bash
pip install -r requirements.txt
python methods/evaluator_retry/run_experiment.py --config configs/evaluator_retry/rare_driving_frog.json
```

Every candidate is saved under `attempts/`; `selected/` contains the final accepted image, or the highest-scoring fallback when the bounded retry budget is exhausted. Scores are relative to the evaluator and must be calibrated on a validation set before cross-prompt claims.
