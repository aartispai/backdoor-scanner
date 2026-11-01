# Backdoor Scanner for Pretrained Models

Multi-stage, trigger-agnostic scanner (white-box weights, no training data, no knowledge of the trigger).
Use case (SDG 9): securing AI visual-inspection models used on manufacturing lines.

## Pipeline
| Stage | Module | Techniques |
|---|---|---|
| 1 Behavioural | `bds/behavioral.py` | random-patch hijack / class-skew sweep, STRIP entropy test |
| 2 Activation | `bds/activation.py` | ABS-style neuron elevation, spectral signatures, activation clustering |
| 3 Trigger search | `bds/trigger.py` | Neural Cleanse mask+pattern optimisation, MAD anomaly index, held-out attack-success validation |
| 4 Aggregation | `bds/aggregate.py`, `bds/report.py` | calibrated logistic regression (or rule-based fallback), explainable JSON/MD report + trigger image |

Stage 3's candidate trigger is cross-checked by STRIP and by the activation tests (stage "Cross-validating candidate trigger").

## Setup
    pip install -r requirements.txt

## 1. Build the zoo and calibrate (this is where false positives get tuned)
    python make_zoo.py --dataset cifar10 --n-clean 8 --n-bd 9 --epochs 8
Trains clean models and BadNets-patch / Blended-backdoored models, scans every one, fits the aggregator,
and prints AUC, FPR/TPR, **per-stage ablations** and **Neural-Cleanse-only / STRIP-only baselines**.
Outputs `zoo/aggregator.joblib` and `zoo/features_eval.json`. Quick dry run: `--dataset synthetic --epochs 3 --steps 60`.

External benchmarks (BackdoorBench / TrojAI): export models as `state_dict` .pt files in a folder with
`labels.json` like `{"m1.pt": {"backdoor": true}, "m2.pt": {"backdoor": false}}`, then
`python make_zoo.py --zoo-dir that_folder --dataset cifar10 --arch resnet18`.

## 2. Scan a model
    python scan.py zoo/bd_9.pt --dataset cifar10          # CLI report in ./reports
    streamlit run app.py                                   # Upload Model -> Report demo

## Notes
- Models load with `torch.load(weights_only=True)`; full-pickle models need `--allow-pickle` (unsafe, can execute code).
- Without `zoo/aggregator.joblib` the scanner falls back to an uncalibrated heuristic and says so in the report.
- Scope: image classifiers. Add architectures in `bds/models.py:build`. Thresholds in `report.py:evidence` are display defaults; the calibrated ensemble is what decides.
- Known limits: input-space triggers only (weight-space/feature-space attacks like WaNet are the next extension); clean models can have natural adversarial patches, which is why evidence is aggregated and calibrated rather than any single detector deciding.
