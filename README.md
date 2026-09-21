# Test-Time Adaptation

## Overview

This repository contains a reproducible test-time adaptation (TTA) method for
pretrained vision models under distribution shift. The released implementation
covers image classification, semantic segmentation, and vision-language
adaptation, and can reach state-of-the-art (SOTA) results under the documented
evaluation protocols.

Datasets, checkpoints, generated results, comparison implementations, and
private research material are intentionally excluded.

## Repository Structure

```text
.
├── code/
│   ├── main.py                       # Classification entry point
│   ├── datasets/                     # Public corruption and shift loaders
│   ├── models/                       # Public backbone loaders
│   ├── utils/                        # Runtime helpers
│   └── segmentation/                 # ACDC segmentation entry point
├── configs/                          # Reproduction YAML configurations
├── scripts/                          # Benchmark and result-aggregation helpers
├── tests/                            # Unit and protocol smoke tests
├── requirements.txt                  # Python dependencies
├── THIRD_PARTY_PROVENANCE.md         # External comparison provenance
└── .gitignore                        # Data, credentials, and private-artifact rules
```

The comparison-method source trees are deliberately absent. Their official
repositories, revisions, licenses, and local wrapper expectations are recorded
in `THIRD_PARTY_PROVENANCE.md` for users who need to reproduce a comparison
table independently.

## Environment

Python 3.10 or newer is recommended. Install a PyTorch build compatible with
your CUDA driver, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

CPU is sufficient for imports, syntax checks, and unit tests. A CUDA GPU is
recommended for benchmark runs.

## Data Preparation

No dataset, checkpoint, API credential, log, or generated result is included.
Download each resource from its official source and place it at the path named
by the selected configuration. The expected dataset locations are:

| Resource | Expected location |
|---|---|
| CIFAR-100-C | `data/cifar-100-c/` |
| ImageNet-C | `data/imagenet-c/` |
| ImageNet-A/R/V2/Sketch | `data/imagenet-a/`, `data/imagenet-r/`, `data/imagenet-v2/`, `data/imagenet-sketch/` |
| ACDC | `data/acdc/` |

## Checkpoint Preparation

Place the required pretrained weights at the paths expected by the selected
configuration:

| Resource | Expected location |
|---|---|
| Cityscapes-pretrained SegFormer-B5 | `models/segformer/segformer-b5-cityscapes/` or the path passed to the segmentation runner |
| CLIP ViT-B/16 | `models/clip/openai/ViT-B-16.pt` |
| CoOp initialization | `models/coop/imagenet/vit_b16_ep50_nctx4_seed1/model.pth.tar-50` |

Do not commit downloaded data or weights. The ignore rules cover common model,
database, cache, credential, and generated-artifact names.

## Quick Smoke Test

```bash
python -m compileall -q code scripts tests
python code/main.py --help
pytest -q tests
bash -n scripts/*.sh scripts/lib/*.sh
```

The smoke tests do not download data or run a benchmark. A one-batch
classification smoke run can be started after the selected dataset and
checkpoint are installed:

```bash
python code/main.py \
  --config configs/vit_imagenetc.yaml \
  --max-batches 1 \
  --gpu 0
```

The default command uses the released adapter. Use `--method source` when a
source-model-only reference is required.

## Reproduce Tables 1-8

The YAML files and shell wrappers define the supported standard, continual,
single-sample, label-shift, mixed-shift, natural-shift, segmentation, and
vision-language protocols. Each wrapper writes JSON summaries below `results/`;
those generated files are ignored by Git.

The following matrix is the shortest route through the eight reported
protocol groups. The wrappers run the source reference and the released
adapter by default for Tables 1-5 and 7-8. Table 6 is different: its wrapper
uses an external vision-language host and an external DEM/AdaDEM checkout,
even when the requested rows are limited to the source reference and the
released adapter. Those checkouts are never committed here.

| Table | Dataset / backbone | Protocol and configuration | Command | Seeds / orders | Expected output and metric | Hardware / runtime |
|---|---|---|---|---|---|---|
| 1 | CIFAR-100-C / ResNeXt-29 | Continual severity-5 stream; `configs/cifar100c_continual.yaml` | `bash scripts/run_cifar100c_continual.sh` | Seed 1997; 3 fixed corruption orders | `results/cifar100c_resnext29_continual/`; top-1 accuracy | CUDA GPU; minutes per seed, hardware dependent |
| 2 | ImageNet-C / GN ResNet-50 and ViT-B/16 | Standard and single-pass continual streams; `configs/imagenetc*.yaml`, `configs/vit_imagenetc*.yaml` | Run `scripts/run_imagenetc_all*.sh`; for continual use `ORDER_IDX=0`, `1`, and `2` with `scripts/run_imagenetc_continual*.sh` | Standard seeds 1997/2048/2077; continual seed 1997 across 3 orders; set `CONTINUAL_REPEATS=50` only for the separate long-horizon protocol | Matching `results/imagenetc_*` directories; mean top-1 accuracy | CUDA GPU with model-appropriate batch size; typically hours for full 15-corruption sweeps |
| 3 | ImageNet-A/R/V2/Sketch / ResNet-50-GN and ViT-B/16 | Natural shifts; corresponding `configs/imagenet_*_vit.yaml` files | `bash scripts/run_natural_shifts.sh` | Seeds 1997/2048/2077 for each shift | `results/imagenet_*`; top-1 accuracy per shift and mean | CUDA GPU; hours for all datasets and seeds |
| 4 | ImageNet-C / ViT-B/16 | Batch-size-one, label-shift, and mixed-shift streams; `configs/vit_imagenetc_wild_*.yaml` | `bash scripts/run_imagenetc_wild_vit.sh` | BS=1 seed 1997; label/mixed-shift seeds 1997/2048/2077 | `results/imagenetc_vit_*_wild*/`; mean top-1 accuracy and per-shift summaries | CUDA GPU; hours, dominated by the batch-size-one stream |
| 5 | ACDC / Cityscapes-pretrained SegFormer-B5 | Ten-round fog/night/rain/snow continual stream; `configs/table5_segformer_acdc.yaml` | `bash scripts/run_table5_segmentation_acdc.sh` | Seed 1997; 10 condition rounds; timestamps 1/4/7/10 | `results/table5_segformer_acdc_surgeon_cotta/`; mIoU and online error | CUDA GPU with substantial memory; hours, depending on sample cap |
| 6 | ImageNet-A/V/R/K / CLIP ViT-B/16 | Episodic prompt-context adaptation; `configs/table6_vlm_tta.yaml`; external host checkouts required | `bash scripts/run_table6_vlm_tta.sh` | Seed 1; zero-shot and CoOp prompt settings; episodic reset per instance | `results/table6_vlm_tta/`; top-1 accuracy by prompt setting and test set | CUDA GPU and CLIP/CoOp weights; hours for all test sets |
| 7 | ImageNet-C / ViT-B/16 | Role-level continual ablations; `configs/vit_imagenetc_continual.yaml` | `bash scripts/ablation_table7.sh` | Default seed 1997; continual order configured by the wrapper | Selected result directory; mean top-1 accuracy and ablation manifest | CUDA GPU; hours for the complete ablation set |
| 8 | ACDC / SegFormer-B5 | Selected-entity normalization ablation; `configs/table5_segformer_acdc.yaml` | `bash scripts/run_table5_sane_ablation.sh` | Seed 1997; 10 condition rounds; timestamps 1/4/7/10 | `results/table5_sane_ablation/`; mIoU and online error | CUDA GPU; hours, depending on sample cap |

For every table, record the YAML revision, checkpoint path, seed/order, batch
size, and any sample cap together with the generated JSON. A one-batch or
small-sample smoke run should be completed before a full sweep. The paper's
multi-seed/order aggregates require repeating the corresponding public runner
with the listed seeds or orders; a default one-seed invocation is not itself a
claim to reproduce an aggregate mean.

### Table 6 external checkouts

The Table 6 wrapper expects user-managed checkouts at the paths below. Pin the
recorded revisions before running and keep both directories ignored by Git:

```bash
git clone https://github.com/TomSheng21/tta-vlm.git code/baselines/tta-vlm-main
git -C code/baselines/tta-vlm-main checkout bcc735fe49cbd2ab5b683781c41c66e1d3f78589
git clone https://github.com/HAIV-Lab/DEM.git code/baselines/DEM-main
git -C code/baselines/DEM-main checkout dee84bf9304fb816c48d9ed8763a8ebf6f902ade
```

The VLM repository does not declare a license file at the pinned revision;
review its terms before use. The public release also omits the project-specific
host modifications used by the historical `source`, `adadem`, and released
adapter rows. A clean upstream checkout is therefore a dependency anchor, not
an assertion of standalone Table 6 reproducibility; the runner stops with a
preflight explanation when those integration entry points are absent.

For a full comparison table, prepare the external repositories listed in
`THIRD_PARTY_PROVENANCE.md` and pass their paths through the documented wrapper
variables. No third-party method source is distributed here.

## Expected Outputs

Each runner writes machine-readable summaries under `results/`, which is
ignored by Git. The expected metric is top-1 accuracy for classification and
vision-language runs, and mIoU plus online error for segmentation. Exact
values depend on the downloaded checkpoint, seed/order, hardware, and any
sample cap recorded in the configuration.

## Baseline Provenance

Comparison implementations are not bundled. Official repositories, tested
revisions where available, license notes, and wrapper expectations are listed
in `THIRD_PARTY_PROVENANCE.md`. Obtain and review each external license before
running a comparison.

## Hardware and Runtime Notes

CPU is sufficient for imports and tests. Full benchmark groups require a CUDA
GPU; ImageNet-C, wild streams, segmentation, and prompt adaptation can take
hours and may require substantial GPU memory. Runtime estimates in the table
are approximate and hardware dependent.

## Troubleshooting

- If a dataset or checkpoint is missing, check the paths in the selected YAML.
- If a comparison import fails, install the corresponding external repository
  and set the wrapper path described in `THIRD_PARTY_PROVENANCE.md`.
- If CUDA memory is insufficient, use the batch-size and sample-cap options
  supported by the selected configuration and record the change in the output
  manifest.
- If a runner stops early, inspect its JSON summary under `results/` before
  rerunning; generated results are local artifacts and are not committed.

## Reproducibility Notes

- Keep the configuration file, random seed, corruption order, and batch size
  fixed when comparing runs.
- Record the exact checkpoint and dataset revisions outside this repository.
- Use the result-aggregation scripts only on locally generated JSON files.
- The public tests validate imports, adapter invariants, and runner argument
  handling; they do not certify a benchmark score.

## License

This public artifact does not include a project license file. Review the
licenses of the code and external assets before redistribution, and contact
the maintainers for project-level licensing terms.

## Citation

Please cite the associated work and this repository when using the released
implementation. Use the bibliographic information supplied by the authors for
the associated work; no paper or private research material is included here.
