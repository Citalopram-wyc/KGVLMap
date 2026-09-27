# Implementation and reproducibility notes

## Model and training preset

The source includes PSA/MGPE/PICR, conditional modality routing, the dense decoder, and fixed UPKG distillation. Swin-B remains frozen in evaluation mode. Frozen RemoteCLIP ViT-B/32 sentence embeddings are cached; the unused visual tower is not kept in the prediction model.

The public single-GPU preset follows the current development run: 1024-pixel crops, 100 epochs, micro-batch 4, gradient accumulation 4, AdamW with learning rate 1e-4 and weight decay 1e-4, three warmup epochs, and cosine decay. This is not the paper's multi-GPU training budget. Reproduction of a published table requires its dataset, split, annotation budget, training configuration, and evaluation checkpoint.

The concrete implementation uses 16 Gaussian bases per scalar attribute, physical encoder pyramid widths 64/128/256/512, 512-dimensional physical/text tokens, 256-channel spatial features, and a four-block decoder with eight heads and window size seven. GroupNorm is used in new CNN blocks. The graph uses nine physical states, restart propagation coefficient 0.85, ten iterations, and smoothing 1.0. These choices are explicit in the code/configuration rather than inferred from a result table.

## Geometry and observation assumptions

Global-instance input preserves complete polygon areas and persistent building IDs. Height and light observations are still sampled from the supplied tile rasters. This does not reconstruct native-grid polygon-intersection-weighted measurements for a full building from unavailable source data. Incomplete observations are conservatively excluded from auxiliary losses.

The polygon-JSON input path is a convenience alternative for existing tile datasets. It derives geometry from per-tile polygons and cannot recover full-scene area or identity when the supplied polygons are clipped. Use the global-instance schema for experiments that require those properties.

UPKG topology refers to a category–attribute graph, not a road network or spatial adjacency graph of buildings. The target's zero background probability contributes a foreground constraint. `foreground` and `uniform` supervision controls keep the same eligible buildings and reliability weighting to examine that contribution.

## Checkpoints and metrics

The training loop saves the most recent complete epoch as `last.pt` and the best validation Macro-F1 checkpoint as `best.pt`. It records optimizer, AMP scaler, and random-number state for epoch-boundary resume. Restore with the same prepared manifest and class definitions. Only load checkpoints from trusted sources.

Evaluation accumulates raw pixel confusion counts over the whole selected split. It does not average tile-level scores or reassign all pixels in a footprint to one function. Diagnostic subset evaluation is marked as partial in its output JSON.

## What is not included

This repository does not redistribute source imagery, building annotations, private file manifests, upstream checkpoints, or finished trained checkpoints. The schema examples are placeholders. No prefilled accuracy tables are shipped as measured output from this release.
