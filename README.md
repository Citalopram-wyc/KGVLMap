# KGVLMap

PyTorch implementation of **KGVLMap: Building function mapping method based on knowledge graph constraints and semantic-guided multimodal adaptive fusion**.

KGVLMap combines RGB imagery, building height, nighttime light, building footprints, and fixed category-level text prompts for pixel-level building function mapping. The default configuration predicts seven building functions and non-building background.

## Architecture

- **Spatial encoders:** a frozen ImageNet-22K Swin-B and three trainable CNNs extract RGB, height, nighttime-light, and footprint-derived morphology features.
- **Physical–Semantic Alignment (PSA):** MGPE encodes building-level height, area, and nighttime-light measurements. Frozen RemoteCLIP text embeddings query these physical tokens to generate an instance-specific condition.
- **PICR:** a training-only regression task predicts the observed normalized nighttime-light value from height and area tokens.
- **Semantic-Driven Modality Routing (SDMR):** the condition generates channel gates and a visual–physical fusion coefficient. Coverage-weighted scattering assembles the fused features on the shared grid.
- **Dense decoder:** four window Transformer blocks and two upsampling stages produce pixel-level predictions.
- **UPKG distillation:** a fixed category–attribute graph supplies reliability-weighted targets for eligible unlabeled training buildings. It is omitted during inference.

The spatial encoders consume rasters; MGPE consumes building-level scalar measurements. They are parallel representations of the observations.

## Installation

Use Python 3.10 or 3.11 and a CUDA-enabled PyTorch installation for training. The tested stack uses PyTorch 2.3.1, torchvision 0.18.1, and timm 1.0.11. Create a separate environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the appropriate PyTorch CUDA build for your system. Training uses one GPU, automatic mixed precision, activation checkpointing, and gradient accumulation. The supplied 100-epoch preset uses micro-batch 4 and accumulation 4, giving an effective batch size of 16. Reduce the micro-batch size if needed.

## Pretrained weights

Download the upstream weights into `weights/`:

```bash
python tools/download_weights.py --output weights
```

The downloader retrieves [Swin-B ImageNet-22K](https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k) and [RemoteCLIP ViT-B/32](https://huggingface.co/chendelong/RemoteCLIP). Both pretrained encoders remain frozen; the text embeddings are cached. Upstream checkpoints are not bundled with this repository.

## Data preparation

See [the data specification](docs/DATA.md) for both supported input formats, class codes, validity masks, and split manifests. Example JSON files in `examples/` illustrate the schema; they are not a dataset.

For a conventional tile dataset, edit `configs/seven_class.yaml` and supply `data/splits.json`. For global building-ID rasters with a fixed 80% annotation budget, edit `configs/seven_class_80pct.yaml` and supply its source manifest. Run all commands from the repository root:

```bash
python prepare_data.py --config configs/seven_class.yaml --workers 4
```

Preparation computes geometry caches, training-only normalization statistics, class weights, and a fixed UPKG. Use `--resume` to continue an interrupted preparation with unchanged inputs and configuration. Completed prepared datasets are protected against overwriting.

## Training, validation, and inference

```bash
# Short end-to-end check using your actual data and pretrained weights.
python train.py --config configs/seven_class.yaml --smoke

# Train; the validation split is evaluated after every epoch.
python train.py --config configs/seven_class.yaml

# Resume from the most recent completed epoch.
python train.py --config configs/seven_class.yaml --resume outputs/runs/seven_class/last.pt

# Evaluate and export georeferenced pixel predictions.
python evaluate.py --config configs/seven_class.yaml --checkpoint outputs/runs/seven_class/best.pt --split val --output outputs/runs/seven_class/evaluation.json --save-predictions
```

Use `--split test` with an independently held-out test split for final evaluation. `best.pt` is selected by validation Macro-F1; `last.pt` supports epoch-boundary resume. An unfinished epoch is repeated after a restart.

Training saves TensorBoard events, `steps.jsonl`, `status.json`, and per-epoch `validation_epoch_XXX.json` files with metrics and raw confusion counts. A local dashboard can display progress and completed validation scores:

```bash
python tools/training_dashboard.py --run-dir outputs/runs/seven_class --port 8877
```

Open `http://127.0.0.1:8877/`. The dashboard binds only to the local computer.

## Evaluation protocol

OA and Cohen's Kappa use the complete eight-class pixel confusion matrix, including background. Macro-Precision, Macro-Recall, and Macro-F1 average the seven functional classes only; class-wise F1 is computed before averaging. Background errors still contribute false positives or false negatives for the affected functions. Ignore pixels are excluded. Predictions are the original pixel-level argmax, without footprint-level majority voting.

## Auxiliary-supervision controls

Set `graph_variant` to `full`, `foreground`, `uniform`, `direct`, or `one_step` to select the UPKG supervision control. The controls retain the reliability weights of the full prior. Set `lambda_topo: 0.0` to disable UPKG and `lambda_picr: 0.0` to disable the PICR loss. Give each experiment a separate `run_dir`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests check coverage-weighted fusion, graph propagation, the foreground/KL decomposition, PICR gradient paths, missing-attribute masking, metrics, raster alignment, annotation hiding, and validation history. They use small generated fixtures and do not require the research dataset or pretrained weights.

## Release scope

This release contains model code, training/evaluation entry points, preprocessing, example configurations, and tests. Dataset rasters, private manifests, pretrained weights, trained checkpoints, and experimental output files are not included. Dataset and checkpoint availability must be established separately; no paper result is claimed from running the included schema examples.

See [implementation notes](docs/IMPLEMENTATION.md) for data-resolution assumptions, implementation choices, and the relationship between the supplied single-GPU preset and the paper. Dependency and weight sources are listed in [third-party notices](THIRD_PARTY_NOTICES.md).
