# Test-Time Adaptation

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

## Installation

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

## Data and Checkpoints

No dataset, checkpoint, API credential, log, or generated result is included.
Download each resource from its official source and place it at the path named
by the selected configuration.

| Resource | Expected location |
|---|---|
| CIFAR-100-C | `data/cifar-100-c/` |
| ImageNet-C | `data/imagenet-c/` |
| ImageNet-A/R/V2/Sketch | `data/imagenet-a/`, `data/imagenet-r/`, `data/imagenet-v2/`, `data/imagenet-sketch/` |
| Cityscapes to ACDC | `data/acdc/` plus a Cityscapes-pretrained SegFormer checkpoint |
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

## Benchmark Protocols

The YAML files and shell wrappers define the supported standard, continual,
single-sample, label-shift, mixed-shift, natural-shift, segmentation, and
vision-language protocols. Each wrapper writes JSON summaries below `results/`;
those generated files are ignored by Git.

The following matrix is the shortest route through the eight reported
protocol groups. The wrappers run the source reference and the released
adapter by default; no comparison source is required for these commands.

| Group | Dataset / backbone | Protocol and configuration | Command | Output and metric | Hardware / runtime |
|---|---|---|---|---|---|
| 1 | CIFAR-100-C / ResNeXt-29 | Continual severity-5 stream; `configs/cifar100c_continual.yaml` | `bash scripts/run_cifar100c_continual.sh` | `results/cifar100c_resnext29_continual/`; top-1 accuracy | CUDA GPU; minutes per seed, hardware dependent |
| 2 | ImageNet-C / GN ResNet-50 and ViT-B/16 | Standard and continual corruption streams; `configs/imagenetc*.yaml`, `configs/vit_imagenetc*.yaml` | Run `scripts/run_imagenetc_all*.sh` and `scripts/run_imagenetc_continual*.sh` | Matching `results/imagenetc_*` directories; mean top-1 accuracy | CUDA GPU with model-appropriate batch size; typically hours for full 15-corruption sweeps |
| 3 | ImageNet-A/R/V2/Sketch / ResNet-50-GN and ViT-B/16 | Natural shifts; corresponding `configs/imagenet_*_vit.yaml` files | `bash scripts/run_natural_shifts.sh` | `results/imagenet_*`; top-1 accuracy per shift and mean | CUDA GPU; hours for all datasets and seeds |
| 4 | ImageNet-C / ViT-B/16 | Batch-size-one, label-shift, and mixed-shift streams; `configs/vit_imagenetc_wild_*.yaml` | `bash scripts/run_imagenetc_wild_vit.sh` | `results/imagenetc_vit_*_wild*/`; mean top-1 accuracy and per-shift summaries | CUDA GPU; hours, dominated by the batch-size-one stream |
| 5 | ACDC / Cityscapes-pretrained SegFormer-B5 | Ten-round fog/night/rain/snow continual stream; `configs/table5_segformer_acdc.yaml` | `bash scripts/run_table5_segmentation_acdc.sh` | `results/table5_segformer_acdc_surgeon_cotta/`; mIoU and online error | CUDA GPU with substantial memory; hours, depending on sample cap |
| 6 | ImageNet-A/V/R/K / CLIP ViT-B/16 | Episodic prompt-context adaptation; `configs/table6_vlm_tta.yaml` | `bash scripts/run_table6_vlm_tta.sh` | `results/table6_vlm_tta/`; top-1 accuracy by prompt setting and test set | CUDA GPU and CLIP/CoOp weights; hours for all test sets |
| 7 | ImageNet-C / ViT-B/16 | Role-level continual ablations; `configs/vit_imagenetc_continual.yaml` | `bash scripts/ablation_table7.sh` | Selected result directory; mean top-1 accuracy and ablation manifest | CUDA GPU; hours for the complete ablation set |
| 8 | ACDC / SegFormer-B5 | Selected-entity normalization ablation; `configs/table5_segformer_acdc.yaml` | `bash scripts/run_table5_sane_ablation.sh` | `results/table5_sane_ablation/`; mIoU and online error | CUDA GPU; hours, depending on sample cap |

For every group, record the YAML revision, checkpoint path, seed, corruption
order, batch size, and any sample cap together with the generated JSON. A
one-batch or small-sample smoke run should be completed before a full sweep.

For a full comparison table, prepare the external repositories listed in
`THIRD_PARTY_PROVENANCE.md` and pass their paths through the documented wrapper
variables. No third-party method source is distributed here.

## Reproducibility Notes

- Keep the configuration file, random seed, corruption order, and batch size
  fixed when comparing runs.
- Record the exact checkpoint and dataset revisions outside this repository.
- Use the result-aggregation scripts only on locally generated JSON files.
- The public tests validate imports, adapter invariants, and runner argument
  handling; they do not certify a benchmark score.

## License and Citation

Check the repository license and the accompanying paper for the applicable
terms. Please cite the associated work when using this implementation.
