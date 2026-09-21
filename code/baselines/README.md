# External comparison location

Comparison-method source code is intentionally not included in this public
release. The released entry points run the source model or the released TTA
adapter without this directory. Optional comparison wrappers accept separate
user-managed checkouts; see `THIRD_PARTY_PROVENANCE.md` for the recorded
repositories, revisions, and licenses.

For the vision-language wrapper, the expected local paths are:

```text
code/baselines/tta-vlm-main/   # upstream VLM-TTA host checkout
code/baselines/DEM-main/       # upstream DEM/AdaDEM checkout
```

These directories are ignored by Git. The public release does not contain the
project-specific integration files that historically extended the VLM host
with the `source`, `adadem`, and released-adapter dispatch entries. The Table 6
runner reports this requirement during preflight instead of silently claiming
that a clean upstream checkout reproduces the complete matrix.
