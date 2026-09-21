# Test-Time Adaptation

This repository provides a reproducible implementation of a test-time
adaptation (TTA) method for pretrained vision models under distribution shift.
The released method is evaluated on image classification, dense prediction,
and vision-language adaptation, and can reach state-of-the-art (SOTA) results
under the documented protocols. Data, checkpoints, comparison implementations,
logs, and private research material are intentionally excluded.

## Repository Structure

```text
.
├── code/
│   ├── atlas/
│   │   ├── common.py                 # Shared views, scores, anchors, and normalization
│   │   ├── classification.py         # Image classification TTA adapter
│   │   ├── segmentation.py           # Dense-prediction TTA adapter
│   │   ├── vlm_instance.py           # Episodic prompt-context adaptation
│   │   ├── vlm_prompt_ensemble.py    # CLIP prompt-view construction
│   │   └── __init__.py
│   ├── datasets/__init__.py          # CIFAR-C/ImageNet-C/natural-shift loaders
│   ├── models/__init__.py            # Public backbone loaders
│   ├── segmentation/
│   │   └── run_acdc_segformer_table5.py
│   ├── utils/
│   │   ├── runtime.py                # Public runner utilities
│   │   ├── fisher.py                 # Optional Fisher helper
│   │   └── __init__.py
│   └── main.py                       # Classification experiment entry point
├── configs/                          # Reproduction YAML files
├── scripts/
│   ├── lib/run_common.sh             # Shared shell helpers
│   ├── run_cifar100c_continual.sh
│   ├── run_imagenetc_all*.sh
│   ├── run_natural_shifts.sh
│   ├── run_imagenetc_wild_vit.sh
│   ├── run_table5_segmentation_acdc.sh
│   ├── run_table6_vlm_tta.sh
│   ├── ablation_table7.sh
│   ├── run_table5_sane_ablation.sh
│   ├── generate_*summary.py          # Result aggregation utilities
│   └── visualization/make_acdc_qualitative.py
├── tests/                             # Unit and protocol smoke tests
├── requirements.txt
├── THIRD_PARTY_PROVENANCE.md
└── .gitignore
```

The comparison-method directories are deliberately absent. Their official
repositories, revisions, licenses, and local wrapper expectations are listed
in `THIRD_PARTY_PROVENANCE.md`; obtain them directly from their authors when a
full comparison table is required.

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

| Resource | Expected location | Notes |
|---|---|---|
| CIFAR-100-C | `data/cifar-100-c/` | Include one `.npy` file per corruption and `labels.npy`. |
| ImageNet-C | `data/imagenet-c/` | ImageNet validation data are also needed for some loaders. |
| ImageNet-A/R/V2/Sketch | `data/imagenet-a/`, `data/imagenet-r/`, `data/imagenet-v2/`, `data/imagenet-sketch/` | Follow the directory layout in the natural-shift YAML files. |
| Cityscapes to ACDC | `data/acdc/` and a Cityscapes-pretrained SegFormer-B5 checkpoint | Both are obtained through their official portals. |
| CLIP ViT-B/16 | `models/clip/openai/ViT-B-16.pt` | Download from the official CLIP release. |
| CoOp initialization | `models/coop/imagenet/vit_b16_ep50_nctx4_seed1/model.pth.tar-50` | Needed only for the CoOp prompt setting. |

Do not commit downloaded data or weights. The ignore rules cover common model,
database, cache, credential, and generated-artifact names.

## Quick Smoke Test

```bash
python -m compileall -q code scripts tests
python code/main.py --help
pytest -q tests
bash -n scripts/*.sh scripts/lib/*.sh
```

The smoke tests do not download data or run a benchmark. To enter the
classification path without third-party comparisons, use an installed dataset
and a one-batch cap, for example:

```bash
python code/main.py \
  --config configs/vit_imagenetc.yaml \
  --method atlas --max-batches 1 --gpu 0
```

## Reproduce the Main Tables

The commands below run the released method. The batch wrappers also enumerate
comparison methods; those rows require the external checkouts documented in
`THIRD_PARTY_PROVENANCE.md`. `SEED`, `GPU`, `MAX_SAMPLES`, and data-root values
can be overridden through the corresponding configuration or environment
variables.

### Table 1: CIFAR-100-C continual

Dataset: CIFAR-100-C, ResNeXt-29, severity 5, three fixed corruption orders.

```bash
python code/main.py --config configs/cifar100c_continual.yaml \
  --method atlas --seed 1997 --gpu 0 \
  --corruption-order brightness,contrast,defocus_blur,elastic_transform,fog,frost,gaussian_noise,glass_blur,impulse_noise,jpeg_compression,motion_blur,pixelate,shot_noise,snow,zoom_blur \
  --order-idx 0
```

Results are written under the configured `logging.results_dir` (normally
`results/cifar100c_resnext29_continual/`). Repeat with the two order strings in
`scripts/run_cifar100c_continual.sh` for the reported order spread.

### Table 2: ImageNet-C standard and continual

The ResNet-50-GN standard/continual configs are `configs/imagenetc.yaml` and
`configs/imagenetc_continual.yaml`. The ViT-B/16 configs are
`configs/vit_imagenetc.yaml` and `configs/vit_imagenetc_continual.yaml`.
Standard evaluation resets before each corruption; continual evaluation keeps
one online state across the stream. The corresponding batch wrappers are
`scripts/run_imagenetc_all.sh`, `scripts/run_imagenetc_continual.sh`,
`scripts/run_imagenetc_all_vit.sh`, and `scripts/run_imagenetc_continual_vit.sh`.

```bash
python code/main.py --config configs/vit_imagenetc.yaml \
  --method atlas --seed 1997 --gpu 0 --max-batches 1
python code/main.py --config configs/vit_imagenetc_continual.yaml \
  --method atlas --seed 1997 --gpu 0 --max-batches 1
```

The one-batch commands are smoke checks. Remove `--max-batches 1` for the full
15-corruption run and repeat the documented seeds/orders.

### Table 3: Natural shifts

The ViT-B/16 configurations are `configs/imagenet_a_vit.yaml`,
`configs/imagenet_r_vit.yaml`, `configs/imagenet_v2_vit.yaml`, and
`configs/imagenet_sketch_vit.yaml`. Run one dataset at a time with
`--method atlas`; `scripts/run_natural_shifts.sh` is the multi-dataset wrapper.
Results are saved in the matching `results/imagenet_*_vit_base_patch16_224/`
directory.

### Table 4: Wild ImageNet-C

This table uses ViT-B/16 with label-shift, mixed-shift, and batch-size-one
protocols. Use the released wrapper for a method-only run:

```bash
bash scripts/run_imagenetc_wild_vit.sh --method atlas --scenario bs1 --seed 1997 --gpu 0
bash scripts/run_imagenetc_wild_vit.sh --method atlas --scenario label_shifts --seed 1997 --gpu 0
bash scripts/run_imagenetc_wild_vit.sh --method atlas --scenario mix_shifts --seed 1997 --gpu 0
```

The three output directories are named in the wrapper and the matching YAML
files are `vit_imagenetc_wild_bs1.yaml`,
`vit_imagenetc_wild_labelshift.yaml`, and
`vit_imagenetc_wild_mixshifts.yaml`.

### Table 5: Cityscapes to ACDC segmentation

Backbone: Cityscapes-pretrained SegFormer-B5. Protocol: batch size 1,
Fog/Night/Rain/Snow stream repeated for 10 rounds, with online error and mIoU.

```bash
METHODS=atlas bash scripts/run_table5_segmentation_acdc.sh
```

Set `ACDC_ROOT`, `SEGFORMER_MODEL`, `GPU`, and `MAX_SAMPLES` before running.
The runner writes JSON summaries under
`results/table5_segformer_acdc_surgeon_cotta/`.

### Table 6: Prompt-only VLM adaptation

Backbone: CLIP ViT-B/16. The protocol is episodic: prompt context and optimizer
state reset for every test instance. Run zero-shot and CoOp initializations
separately:

```bash
METHODS=atlas PROMPT_SETTINGS=zs bash scripts/run_table6_vlm_tta.sh --method atlas --mode zs
METHODS=atlas PROMPT_SETTINGS=coop bash scripts/run_table6_vlm_tta.sh --method atlas --mode coop
```

Set `COOP_CKPT` for the CoOp run. Outputs are stored under
`results/table6_vlm_tta/`.

### Table 7: Component ablations

The wrapper runs the released method's role-level ablations on continual
ImageNet-C ViT-B/16. It does not require comparison source code for the
method-only rows, but the shell environment must provide the configured Python
dependencies.

```bash
bash scripts/ablation_table7.sh results/table7_release
```

### Table 8: Selected-entity normalization ablation

```bash
bash scripts/run_table5_sane_ablation.sh
```

This uses the Cityscapes to ACDC segmentation protocol and writes the summary
to `results/table5_sane_ablation/`.

## Expected Outputs

Each runner writes JSON result records with the dataset, method, protocol,
seed/order, per-condition metrics, and aggregate accuracy or mIoU. Summary
CSV/Markdown files are generated alongside the JSON records when the full
wrapper is used. Exact values depend on the downloaded checkpoint versions,
hardware, and the documented seed/order protocol.

## Comparison-Method Provenance

Comparison implementations are not copied into this repository. Use the
official URLs and revisions in `THIRD_PARTY_PROVENANCE.md`, inspect their
current licenses, and keep their checkouts outside this repository. A missing
or incompatible comparison implementation should be reported as unavailable,
not replaced with an unverified local implementation.

## Hardware and Runtime Notes

Full ImageNet-C and VLM runs require a CUDA GPU and substantial disk space for
the datasets and checkpoints. Segmentation and prompt adaptation have separate
environment requirements documented in their YAML files and runner messages.
Use `--max-batches 1` or the runner's sample cap to validate paths before a
long evaluation.

## Troubleshooting

- `FileNotFoundError`: check the configured data or checkpoint path and confirm
  that the official resource has been downloaded.
- Missing comparison module: obtain the external checkout listed in
  `THIRD_PARTY_PROVENANCE.md`, or run only the released method.
- CUDA out of memory: lower the configured batch size and record the change in
  the result metadata.
- Reproducibility differences: verify the checkpoint revision, seed, corruption
  order, reset policy, and package versions before comparing numbers.

## License

The project files in this repository are released under the repository license.
Third-party datasets, model weights, and comparison implementations remain
under their own licenses and are not relicensed here.

## Citation

Please cite the associated method publication and the original dataset,
backbone, and comparison-method papers when using this repository. The public
release intentionally keeps this README focused on the implementation and
reproduction instructions.

## Security and Privacy

Never commit API keys, access tokens, passwords, SSH keys, private paths,
personal metadata, experiment logs, unpublished plans, submission material,
or internal analysis. Review `git status` and the staged file list before every
push. The `.gitignore` is intentionally conservative, but it is not a
substitute for reviewing the files that will be published.
