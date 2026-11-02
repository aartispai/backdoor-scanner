# Backdoor Scanner for Pretrained Models

Multi-stage, trigger-agnostic scanner (white-box weights, no training data, no knowledge of the trigger).
Use case (SDG 9): securing AI visual-inspection models used on manufacturing lines.

## Pipeline
| Stage | Module | Techniques |
|---|---|---|
| 1 Behavioural | `bds/behavioral.py` | random-patch hijack / class-skew sweep, STRIP entropy test |
| 2 Activation | `bds/activation.py` | ABS-style neuron elevation, spectral signatures, activation clustering |
| 3 Trigger search | `bds/trigger.py` | Neural Cleanse mask+pattern optimisation, MAD anomaly index, **attack success at a fixed 3% pixel budget vs the median class** (`nc_asr_gap`) |
| 4 Aggregation | `bds/aggregate.py`, `bds/report.py` | calibrated logistic regression (or rule-based fallback), explainable JSON/MD report + trigger image |

Stage 3's candidate trigger is cross-checked by STRIP and by the activation tests (stage "Cross-validating candidate trigger").

## Setup
    pip install -r requirements.txt

## 1. Build the zoo and calibrate (this is where false positives get tuned)
    python make_zoo.py --dataset cifar10 --n-train 50000 --epochs 15 --out zoo_cifar          # easy zoo
    python make_zoo.py --dataset cifar10 --n-train 50000 --epochs 15 --hard \
           --archs smallcnn,resnet18 --attacks patch,blend,warp --n-clean 12 --n-bd 18 --out zoo_hard
`--hard` adds low poison rates (2-10%), 2-4 px patches and **hard-negative clean models** (augmented, fine-tuned);
`warp` is a WaNet-style non-patch trigger (may be dropped automatically if the backdoor does not implant).
Use a separate `--out` folder per run. Runs are resumable and scan features are cached.

Printed evaluation (also saved in `<out>/features_eval.json`):
- cross-validated ensemble AUC (averaged over 5 shuffles), FPR/TPR at 0.5
- **TPR at FPR<=0%/5%** plus the score threshold that achieves it
- ablations: per stage group **and per single feature**, plus single-feature AUCs and Neural-Cleanse-only / STRIP-only baselines
- **leave-one-attack-out**: train on some attack types, test on the held-out one (the real test of trigger-agnostic detection)

Changing detector features? Bump `FEAT_VERSION` in `make_zoo.py` so cached scans are recomputed, and re-run to refit `aggregator.joblib`.

External benchmarks (BackdoorBench / TrojAI): export models as `state_dict` .pt files in a folder with
`labels.json` like `{"m1.pt": {"backdoor": true, "attack": "badnets", "arch": "resnet18"}, "m2.pt": {"backdoor": false}}`, then
`python make_zoo.py --zoo-dir that_folder --dataset cifar10`.

## 2. Scan a model
    python scan.py zoo/bd_9.pt --dataset cifar10          # CLI report in ./reports
    streamlit run app.py                                   # Upload Model -> Report demo

## Notes
- Models load with `torch.load(weights_only=True)`; full-pickle models need `--allow-pickle` (unsafe, can execute code).
- Without `zoo/aggregator.joblib` the scanner falls back to an uncalibrated heuristic and says so in the report.
- Scope: image classifiers. Add architectures in `bds/models.py:build`. Thresholds in `report.py:evidence` are display defaults; the calibrated ensemble is what decides.
- Known limits: input-space triggers only (weight-space/feature-space attacks like WaNet are the next extension); clean models can have natural adversarial patches, which is why evidence is aggregated and calibrated rather than any single detector deciding.
