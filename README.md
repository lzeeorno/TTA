# Test-Time Adaptation

This repository provides a research implementation of a test-time adaptation
(TTA) method for adapting pretrained vision models to distribution shifts
without target labels. In the evaluated settings, the method can reach
state-of-the-art (SOTA) performance across classification, dense prediction,
and vision-language adaptation tasks.

## Highlights

- Reliable evidence selection from multiple predictions and views.
- Anchored online updates that reduce cumulative adaptation drift.
- Entity-aware normalization for images, pixels, and prompt views.
- Support for standard, continual, wild-stream, segmentation, and
  vision-language TTA workflows.
- Reproducible configurations, runners, summaries, visualizations, and tests.

## Project Structure

```text
.
├── code/
│   ├── core adaptation modules/
│   ├── data and model loaders/
│   ├── segmentation and vision-language runners/
│   ├── auxiliary adaptation components/
│   └── comparison-method placeholder/
├── configs/
│   └── YAML evaluation configurations
├── scripts/
│   ├── benchmark runners/
│   ├── result-summary utilities/
│   └── visualization utilities/
├── tests/
│   └── unit and protocol tests
├── requirements.txt
├── THIRD_PARTY_PROVENANCE.md
└── .gitignore
```

Comparison-method source code is intentionally not redistributed. External
comparison implementations, datasets, and model checkpoints must be obtained
from their respective official sources and used under their own licenses.

## Installation

Python 3.10 or newer is recommended. Install a PyTorch build compatible with
your hardware, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

CUDA is recommended for full-scale evaluation. CPU execution is sufficient
for static checks and the unit tests.

## Quick Check

```bash
python -m compileall -q code scripts tests
pytest -q tests
```

The tests do not download datasets or checkpoints. A small number of optional
protocol tests are skipped when an external vision-language dependency is not
available.

## Data and Checkpoints

Datasets and pretrained weights are not included. Download them directly from
the official project pages, review their terms, and place them in the paths
specified by the selected configuration. Generated results, logs, caches,
checkpoints, databases, credentials, and private development material are
ignored by Git.

## Running Experiments

Select a configuration under `configs/` and run the corresponding shell
script under `scripts/`. Each runner documents its expected data paths,
output directory, protocol, and resource requirements. Start with a small
sample or a single stream when validating a new environment, then launch the
full evaluation after the smoke checks pass.

## Reproducibility and Attribution

The repository contains the implementation and public evaluation utilities.
Please cite the associated method work when using this code, and cite any
external datasets, backbones, or comparison methods separately. No third-party
source code or model weights are relicensed by this repository.

## Security

Do not commit API keys, access tokens, passwords, private keys, personal
metadata, local databases, experiment logs, or unpublished planning material.
The repository ignore rules cover these categories, but review `git status`
and the staged file list before every push.
