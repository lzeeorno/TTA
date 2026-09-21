# Third-party provenance

This release does **not** redistribute any comparison-method source code.
The table below records the external methods used by the historical runners so
that a researcher can obtain them directly from the official repositories and
review their current licenses. The release itself contains no third-party
baseline files, nested `.git` directories, datasets, checkpoints, or logs.

| Method | Official repository | Local revision | License / redistribution status | Local wrapper |
|---|---|---|---|---|
| Tent | https://github.com/DequanWang/tent | `e9e926a668d85244c66a6d5c006efbd2b82e83e8` | MIT; external checkout required | historical `code/main.py` import |
| EATA | https://github.com/mr-eggplant/EATA | `f739b3668cc7617e9b9f1979c1a358497a3472c3` | MIT; external checkout required | historical `code/main.py` import |
| SAR | https://github.com/mr-eggplant/SAR | `20f6e24b17525f34503510afccedc0629b67b7c4` | BSD-3-Clause; external checkout required | historical `code/main.py` import |
| CoTTA | https://github.com/qinenergy/cotta | `c212a204b32be4005092e4323105a24a29ad2952` | MIT; external checkout required | historical `code/main.py` import |
| DeYO | https://github.com/Jhyun17/DeYO | `3d7d2897d864907571ef75edf31f0028d53f1432` | MIT; external checkout required | historical `code/main.py` import |
| RoTTA | https://github.com/BIT-DA/RoTTA | `67e34c900cdd355fc07e55edd4c577ea7b8ebcc9` | MIT; external checkout required | historical `code/main.py` import |
| FOA | https://github.com/mr-eggplant/FOA | local snapshot has no usable Git revision | NTUITIVE non-commercial license has a conflicting no-distribution clause; external checkout only | historical `code/main.py` import |
| SURGEON | https://github.com/chenjoya/SURGEON | local snapshot has no usable Git revision | No license file in inspected snapshot; external checkout only | historical `code/main.py` import |
| VLM-TTA | https://github.com/mala-lab/VLM-TTA | local snapshot has no usable Git revision | No license file in inspected snapshot; external checkout only | `code/atlas/vlm_instance.py` import path |

For all comparison methods, clone the official repository yourself after
checking its current license and adapt the local import paths as needed. The
public release does not grant redistribution rights for those sources.
