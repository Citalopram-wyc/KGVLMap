# Data specification

Run entry points from the repository root. Paths in the example manifests are relative to that directory. Replace them with your own relative or absolute paths; the examples do not include any imagery or annotations.

## Class codes

The public presets use this order:

```text
0  Background
1  Residential
2  Commercial
3  Administrative
4  Healthcare
5  Educational
6  Sport and art
7  Industrial
8  Building with a hidden functional label (input annotation only)
255 Missing/invalid imagery
```

The network outputs eight classes, numbered 0–7. Code 8 is not an output class. `source_label_map` converts raw codes 8 and 255 to segmentation ignore value 255, but the instance raster and validity mask preserve the difference: unlabeled buildings retain geometry and physical observations, while missing imagery does not contain an eligible building instance.

Keep `classes`, `prompts`, the raw-to-training code mapping, and the source manifest class order consistent. If your source dataset uses a different code order, explicitly change the mapping rather than guessing from colors.

## Option A: tile rasters and polygon JSON

The conventional folder layout is:

```text
data/buildings/
  img/train_001.tif       # 3-band RGB, uint8
  BH/train_001.tif        # 1-band building height
  NL/train_001.tif        # nighttime light; channels must match configuration
  label/train_001.tif     # 1-band integer labels
  json_en/train_001.json  # footprint polygons in pixel coordinates
```

Every tile uses the same filename across raster folders, with the matching stem for JSON. `examples/footprint.json` demonstrates the LabelMe-style `shapes` format. ISAT-style `objects` with `segmentation` coordinates are also supported. Functional labels and descriptions stored in polygon JSON are ignored: supervised classes come only from the label raster.

All rasters need valid georeferencing. RGB defines the reference grid. Physical rasters are aligned with bilinear resampling and label rasters with nearest-neighbor resampling. Polygon coordinates refer to the RGB grid unless an explicit `geometry_reference_path` is supplied through a source manifest.

Use `configs/seven_class.yaml`. Copy and replace `examples/splits.json` as `data/splits.json`. Split entries are tile stems, without `.tif`. The reader excludes training tiles that intersect validation/test bounds plus the configured buffer. The automatic spatial-block split is available by setting `split_manifest: null`; it is not a substitute for the paper's cross-city protocol.

Polygon JSON mode does not preserve city-wide building identity across tiles. For annotation-budget experiments with buildings appearing in multiple tiles, use Option B.

## Option B: explicit manifest and global instance rasters

`examples/source_manifest.json` demonstrates the record structure. Each record contains a unique tile `id`, a `split`, and paths to:

- `rgb`, `height`, `ntl`, and `label` rasters;
- `instance_id`: an integer raster with background 0 and positive, stable global building IDs;
- `valid_mask`: a raster with 1 for observed imagery and 0 for missing imagery;
- `instances`: a JSON list of `building_id` and complete-polygon `full_area_m2`, as in `examples/instances.json`.

Set `geometry_format: instance_raster`. Instance IDs and validity masks must exactly match the RGB CRS, transform, shape, and single-band format. Instance IDs are zero where imagery is missing. The metadata must contain exactly the positive IDs present in that tile. Complete footprint area is expressed in square meters, excluding interior holes.

For a fixed 80/20 annotation budget, use `configs/seven_class_80pct.yaml`. Its source manifest must declare `annotation_policy.visible_fraction: 0.8` and `annotation_policy.seed: 42`. All occurrences of the same selected training building must have their function hidden together as raw code 8. Validation and test labels are never hidden. Verify spatial separation and global-ID separation before preparing caches: explicit manifest splits are used as supplied.

The specialized `tools/create_annotation_budget.py` can recreate the author's 80/20 protocol from an already rasterized and audited annotation export. It requires `source_snapshot.json`, `identity_audit.json`, and `aligned_sources.jsonl`; each source row includes `existing_split`, `nominal_split`, `district`, and the paths above. Its instance metadata additionally includes `function_code`, which is used only to create the annotation budget. This export and its audit files are not distributed here. For your own source pipeline, generate an equivalent manifest and masks directly.

## Physical observations

Height and nighttime-light statistics are extracted within footprint ownership masks. The image validity mask excludes missing observations. Buildings that touch tile boundaries or invalid-image boundaries are retained for conditioning but excluded from auxiliary supervision and attribute-statistics fitting when their measurements are incomplete.

The current presets expect three-channel nighttime-light imagery and use `ntl_scalar: luminance_rgb`, a Rec.709 brightness proxy. For a calibrated single-band product, set `ntl_channels: 1` and `ntl_scalar: first_band`. Never interpret rendered RGB brightness as calibrated radiance.

Training-only normalization uses log-transformed scalar attributes and the 1st/99th percentiles. The UPKG associations use only visible training functions. Hidden ground-truth diagnostic files are for post-training inspection, not teacher construction, checkpoint selection, or hyperparameter tuning.
