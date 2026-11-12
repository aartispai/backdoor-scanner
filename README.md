# Backdoor Scanner for Image Classifiers

A research prototype for inspecting image-classification models for possible backdoors. It combines behavioral tests, activation analysis, and trigger search, then aggregates their evidence into a verdict and report.

The scanner analyzes a model using a clean reference dataset; it does not need the model's original training set or prior knowledge of its trigger. Results are indicators for investigation, not a guarantee that a model is safe.

## How it works

| Stage | Module | What it checks |
|---|---|---|
| Behavioral tests | `bds/behavioral.py` | Whether random patches cause class-skew and whether a candidate trigger survives STRIP-style input blending. |
| Activation analysis | `bds/activation.py` | Neuron elevation and whether activations for triggered inputs stand out from genuine inputs. |
| Trigger search | `bds/trigger.py` | Optimizes candidate triggers for each target class, then compares mask anomalies and attack success against other classes. |
| Evidence aggregation | `bds/aggregate.py` | Combines the detector features into a backdoor score using a zoo-trained logistic-regression model, or a built-in heuristic if no aggregator is available. |
| Reporting | `bds/report.py` | Produces a verdict, per-detector evidence, reconstructed-trigger image, and JSON/Markdown reports. |

The scanner gathers evidence; the aggregator scores that evidence. The aggregator does not inspect model weights or search for triggers by itself.

## Requirements and setup

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The application supports CIFAR-10, GTSRB, and a synthetic dataset. CIFAR-10 and GTSRB are downloaded automatically by the dataset loader when first used. The datasets are stored under `data/`, which is excluded from Git.

## Build a model zoo and calibrate the scanner

Build a default CIFAR-10 zoo:

```powershell
python make_zoo.py --dataset cifar10 --out zoo
```

This trains clean and backdoored sample models, scans them, and fits the evidence aggregator. Backdoored models that do not reach the required attack success rate are retrained and may be dropped. By default, the script uses 8 clean models, 9 backdoored models, 8 training epochs, 10,000 training examples, 500 scan examples, and 200 trigger-search steps.

For a more varied zoo with multiple architectures, attacks, and harder clean examples:

```powershell
python make_zoo.py --dataset cifar10 --hard `
  --archs smallcnn,resnet18 --attacks patch,blend,warp `
  --n-clean 12 --n-bd 18 --out zoo_hard
```

The zoo directory contains model weights and metadata, plus:

- `aggregator.joblib` — fitted model used to combine scanner evidence.
- `features_eval.json` — evaluation metrics for that zoo.
- `features_cache.json` — cached scan features so reruns can reuse completed scans.
- `labels.json` — model labels and training metadata.

Runs can resume from saved models and cached features. Use a separate output directory for each experiment. Evaluation metrics depend on the models and attacks in the zoo; they are not a guarantee of performance on every real-world model.

To use an external zoo, provide model `state_dict` files and a `labels.json` in the same directory:

```json
{
  "model_a.pt": {"backdoor": true, "attack": "badnets", "arch": "resnet18"},
  "model_b.pt": {"backdoor": false, "arch": "smallcnn"}
}
```

Then run:

```powershell
python make_zoo.py --zoo-dir path\to\external-zoo --dataset cifar10
```

Each entry must include `backdoor`; `arch` is optional and defaults to the `--arch` value (which defaults to `smallcnn`). The model files must be compatible state dictionaries.

## Scan a model from the command line

```powershell
python scan.py zoo\bd_8.pt --dataset cifar10 --arch smallcnn `
  --aggregator zoo\aggregator.joblib
```

The command scans the selected model using clean reference examples, prints a verdict and detector evidence, and writes reports to `reports/` by default. If the aggregator file is missing, it uses an **uncalibrated heuristic** instead. You can set `--out`, `--n-clean`, and `--steps` to change the report directory, number of clean reference examples, and trigger-search effort. Run `python scan.py --help` for all options.

The default dataset is `cifar10`, the default architecture is `smallcnn`, and the default aggregator path is `zoo/aggregator.joblib`.

## Scan a Hugging Face model

```powershell
python scan.py --hf aaraki/vit-base-patch16-224-in21k-finetuned-cifar10 --dataset cifar10 --aggregator zoo_cifarggregator.joblib
```

Works with `transformers` image-classification models (Hub id or local folder). Weights are loaded from **safetensors only** with `trust_remote_code` off, so scanning an untrusted repo cannot run its code; pickle-only repos are refused. Use `--revision <commit>` to pin a version.

- Reference images are resized to the model's input size (usually 224px). For models whose classes differ from `--dataset` (e.g. ImageNet), pass `--images path	o\clean_images`.
- Models with more than 50 classes search a subset (the most-predicted classes plus a random sample); override with `--max-classes`.
- Clean accuracy is reported as n/a (label order is not matched).
- **Speed:** bf16 mixed precision is automatic on CUDA for inputs larger than 64px (`--amp` / `--no-amp` to force). Trigger-search results are cached per class in `.bds_cache/` (keyed by model weights, reference images and settings), so re-scans are faster and interrupted scans resume; `--no-cache` disables it.
- **`--early-exit` (opt-in):** if the random-patch and neuron-elevation tests are both clearly quiet, the full scan is skipped and the report says only a quick screen ran. On the 64 CNN zoo models those two tests overlap heavily between clean and backdoored models, so the thresholds (0.20, 1.5) skip about 9 of 28 clean models but also 1 of 36 backdoored ones. Not validated on large transformers.
- **Transformer-aware checks** (models with MLP blocks, e.g. ViT): neuron elevation is run on the last block's MLP hidden units instead of conv channels, and an attention diagnostic reports how much CLS attention lands on the reconstructed trigger. On a small set of toy ViTs (`python make_vit_demo.py`, 2 clean / 4 backdoored) the calibrated score separated the classes (backdoored 0.93-0.96, clean 0.06-0.22), but the attention diagnostic did **not**: it was high on a clean model and about 1x on backdoored ones, so it is shown for inspection and never counts as fired.
- Caveats: the aggregator was calibrated on small 32px CNNs, so scores on large transformers are rough guidance. Neuron elevation samples 128 channels of the last convolution (the patch embedding in a ViT). A 224px scan on a laptop GPU takes minutes.

## Try it in Streamlit

```powershell
streamlit run app.py
```

Choose the reference dataset and architecture, leave the aggregator path as `zoo/aggregator.joblib` (or enter another path), and upload a `.pt` or `.pth` model state dictionary. To try one of the sample models in `zoo/`, select that file in the upload control. The app does not automatically choose a model from the zoo.

For a calibrated score, generate and keep `zoo/aggregator.joblib` alongside the sample models. If it is absent, the app displays a score from the uncalibrated heuristic. The Streamlit app displays the report and lets you download its JSON data.

## Repository contents

- `app.py` — Streamlit interface.
- `scan.py` — command-line scanner.
- `make_zoo.py` — model-zoo generation, feature extraction, evaluation, and aggregator calibration.
- `bds/` — scanner stages, model/dataset helpers, aggregation, and reporting.
- `zoo/` — sample model weights for trying scans; `make_zoo.py` generates the aggregator and other zoo outputs.
- `requirements.txt` — Python dependencies.

The root `.gitignore` excludes local environments, caches, datasets, and experiment output directories. The `zoo/` sample models are intentionally not ignored so they can be included in the repository. Review the staged files before pushing, especially if you add larger datasets or generated experiment directories.

## Safety and limitations

- Only use models and aggregators from sources you trust. Model state dictionaries are loaded with `weights_only=True`; the CLI's `--allow-pickle` option is unsafe for untrusted full-model pickle files.
- `aggregator.joblib` is a serialized Python object. Only load it from a trusted source.
- Backdoored sample models are intentionally poisoned test artifacts. Label them clearly and do not deploy them.
- This project targets image classifiers and input-space triggers. Detection is imperfect: clean models may respond strongly to natural patches, and some backdoors may evade these tests. Treat results as screening evidence and review suspicious models further.
