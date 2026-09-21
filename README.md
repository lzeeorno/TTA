# ATLAS test-time adaptation reproduction

This repository is a sanitized public release of the ATLAS implementation
used to generate the eight main tables in the accompanying NeurIPS 2026
manuscript. It preserves the internal identifier `atlas` in code, CLI flags,
configuration files, and result names for reproducibility. The paper display
name is intentionally kept separate from that internal identifier and may be
updated in a later revision.

## 1. Overview

The release contains the ATLAS classification, segmentation, and episodic VLM
entry points, the table-specific configurations, and paper source. Comparison
method source is deliberately absent; it is neither uploaded nor relicensed
here. Data, checkpoints, logs, search history, and private development notes
are excluded.

## 2. Repository Structure

```text
code/                 ATLAS, TRIAD, data/model loaders, and table runners
configs/              only configs referenced by the table commands
scripts/              table runners, summaries, smoke helpers, visualization
tests/                unit and protocol tests
paper/NeurIPS2026_V9/ paper source, tables, figures, and bibliography
code/baselines/       README only; comparison methods are external
```

## 3. Environment

Use Python 3.10 or newer. A CUDA installation matching the selected PyTorch
build is recommended for full tables.

```bash
conda create -n atlas-tta python=3.10 -y
conda activate atlas-tta
pip install -r requirements.txt
```

Table 5 additionally requires `transformers`; Table 6 requires the external
VLM-TTA/CLIP checkout described in `THIRD_PARTY_PROVENANCE.md`.

## 4. Data Preparation

The runners expect the following local directories. Download each dataset from
its official source and accept its terms before use:

| Dataset | Expected path |
|---|---|
| CIFAR-100-C | `data/cifar100-c/CIFAR-100-C` |
| ImageNet-C | `data/imagenet-c` |
| ImageNet validation | `data/imagenet/val` |
| ImageNet-A/R/V2/Sketch | paths in the corresponding `configs/imagenet_*_vit.yaml` |
| ACDC | `data/acdc` |
| VLM datasets | paths in `configs/table6_vlm_tta.yaml` |

No dataset is redistributed in this repository.

## 5. Checkpoint Preparation

Do not commit checkpoints. The expected local paths are:

- `models/clip/openai/ViT-B-16.pt` for Table 6;
- `models/coop/imagenet/vit_b16_ep50_nctx4_seed1/model.pth.tar-50` for CoOp;
- `models/segformer/segformer-b5-cityscapes` for Table 5;
- RobustBench/timm weights resolved by `code/models/__init__.py` for
  classification tables.

Use the upstream download instructions and verify checksums where available.

## 6. Quick Smoke Test

Run static checks first:

```bash
python -m py_compile code/main.py code/atlas/*.py code/segmentation/*.py
python code/main.py --help
pytest -q tests
```

The ATLAS module smoke can run without any comparison-method checkout:

```bash
PYTHONPATH=code python -c "from atlas import ATLAS, ATLASSegmentationAdapter; print('ATLAS core imports: ok')"
```

The historical unified classification runner imports comparison adapters at
startup. Use it only after separately obtaining the required upstream sources;
the public release does not contain those methods. The VLM `ATLASInstance`
entry point likewise requires the external VLM-TTA/DEM checkout described in
`THIRD_PARTY_PROVENANCE.md`.

## 7. Reproduce Table 1

Dataset: CIFAR-100-C. Backbone: ResNeXt-29. Protocol: standard and continual
corruption streams. Config: `configs/cifar100c.yaml` or
`configs/cifar100c_continual.yaml`. Command:

```bash
bash scripts/run_cifar100c_all.sh
bash scripts/run_cifar100c_continual.sh
```

Seeds/orders: the scripts declare the canonical seeds and three continual
orders. Expected output path: `results/`. Expected paper metric: Table 1
accuracy in `paper/NeurIPS2026_V9/tables/table1*.tex`. Runtime depends on the
GPU; a 24 GB GPU is recommended.

## 8. Reproduce Table 2

Dataset: ImageNet-C. Backbones: ResNet-50-GN and ViT-B/16. Protocol: standard
and continual streams. Configs: `configs/imagenetc*.yaml` and
`configs/vit_imagenetc*.yaml`.

```bash
bash scripts/run_imagenetc_all.sh
bash scripts/run_imagenetc_continual.sh
bash scripts/run_imagenetc_all_vit.sh
bash scripts/run_imagenetc_continual_vit.sh
```

Seeds/orders, output paths, expected metrics, and runtime are emitted by each
script and correspond to `table2_imagenetc.tex`.

## 9. Reproduce Table 3

Dataset: ImageNet-A/R/V2/Sketch. Backbone: ViT-B/16. Protocol: natural-shift
evaluation with the per-dataset configs. Command:

```bash
bash scripts/run_natural_shifts.sh
```

The summary is generated under `results/` and maps to
`table3_natural_shift.tex`. The script records the seed and dataset order.

## 10. Reproduce Table 4

Dataset: Wild ImageNet-C. Backbone: ViT-B/16. Protocol: batch-size-one,
label-shift, and mixed-shift streams. Configs:
`configs/vit_imagenetc_wild_*.yaml`.

```bash
bash scripts/run_imagenetc_wild_vit.sh
```

Output summaries are written below `results/` and correspond to
`table4_wild.tex`. These runs require substantially more time than the smoke
test because they process long streams.

## 11. Reproduce Table 5

Dataset: Cityscapes-to-ACDC. Backbone: SegFormer-B5. Protocol: ten-round
continual stream, batch size 1, with the configured timestamp reports.
Config: `configs/table5_segformer_acdc.yaml`.

```bash
METHODS=atlas bash scripts/run_table5_segmentation_acdc.sh
```

Expected output is `results/table5_segformer_acdc_surgeon_cotta/`; metrics map
to `table5_segmentation.tex`. External segmentation baselines are not
redistributed; use the status registry written by the runner.

## 12. Reproduce Table 6

Dataset: ImageNet-A/R/V2/Sketch. Backbone: CLIP ViT-B/16. Protocol: episodic
prompt-only adaptation, separately for zero-shot and CoOp prompt settings.
Config: `configs/table6_vlm_tta.yaml`.

```bash
bash scripts/run_table6_vlm_tta.sh --method atlas --mode zs
bash scripts/run_table6_vlm_tta.sh --method atlas --mode coop
```

Expected output is `results/table6_vlm_tta/`; metrics map to
`table6_vitb16_vlm.tex`. The VLM-TTA source is an external, user-fetched
dependency because its inspected snapshot had no redistributable license.

## 13. Reproduce Table 7

Dataset: ImageNet-C continual. Backbone: ViT-B/16. Protocol: CORE/CARE and
same-role replacement ablations. Config:
`configs/vit_imagenetc_continual.yaml`.

```bash
bash scripts/ablation_table7.sh
```

The script writes a manifest and summary under its selected results directory;
the corresponding LaTeX is `table7_ablation.tex`.

## 14. Reproduce Table 8

Dataset: ACDC segmentation. Backbone: SegFormer-B5. Protocol: selected-entity
versus total-pixel normalization. Config: `configs/table5_segformer_acdc.yaml`.

```bash
bash scripts/run_table5_sane_ablation.sh
```

The summary maps to `table5_sane_placeholder.tex` and the SANE discussion in
the appendix. Use the existing reference result only when its provenance is
available; the script records whether a result is materialized or rerun.

## 15. Expected Outputs

Runners write JSON summaries, per-condition metrics, and optional visual
artifacts beneath `results/`. Generated outputs are intentionally ignored by
Git. The paper tables are the source of the reported values; a new run must
not silently overwrite a formally released result.

## 16. Baseline Provenance

See `THIRD_PARTY_PROVENANCE.md`. No comparison-method source is copied into
this repository; unsupported or unavailable baselines are reported as
external/unavailable.

## 17. Hardware / Runtime Notes

The reported experiments used CUDA and a high-memory GPU. Table 2--4 full
streams are multi-hour jobs; Table 5/6 also depend on checkpoint download and
storage bandwidth. Start with one batch or a small sample limit where the
runner exposes it before launching a full stream.

## 18. Troubleshooting

- Missing data/checkpoint: verify the path in the selected YAML before changing
  code.
- Missing baseline: clone the official source and keep it outside the
  committed tree, then expose it at the path expected by the historical
  runner.
- CUDA out of memory: lower the CLI batch size only for a diagnostic run; do
  not use the diagnostic run as the paper result.
- Mismatched numbers: inspect JSON provenance and seed/order before rerunning.

## 19. License

ATLAS project files are released for research use subject to the repository
owner's publication terms. No third-party baseline source is included. See
`THIRD_PARTY_PROVENANCE.md`; no license is granted for third-party source,
datasets, or checkpoints.

## 20. Citation

Use the citation information in `paper/NeurIPS2026_V9/references.bib` and the
final manuscript. The internal code identifier remains `atlas` so that
published experiment commands and historical result paths stay reproducible.
