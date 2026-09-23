# UnifiedDex

Real-robot five-finger manipulation data from several datasets, converted into one format with a shared
cross-embodiment hand geometry (5 fingertips in the hand's own palm frame). Project rules: `CLAUDE.md`.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv numpy scipy pandas pyarrow h5py av opencv-python-headless huggingface_hub \
    yourdfpy pyyaml matplotlib trimesh gdown fsspec aiohttp pytest
```

Hand models / reference code are cloned into `external_hand_models/` and `external_code/` (see `HANDS.md` §18;
additionally `unitreerobotics/xr_teleoperate` (sparse: `assets/inspire_hand`), `snuvclab/HRDexDB`,
`dexwild/dexwild`, `dexwild/dexwild-training`). HRDexDB robot URDFs come with the dataset download.
Third-party / proprietary model files are git-ignored and must not be redistributed.

## Pipeline

```bash
# 1. samples (raw_data/)
.venv/bin/python scripts/download/hrdexdb_sample.py 10
.venv/bin/python scripts/download/humanoid_everyday_h1_sample.py 20
.venv/bin/python scripts/download/dexwild_robot_sample.py pour 15 2   # remote HDF5 inside tar, range requests
.venv/bin/python scripts/download/realdex_sample.py camera_param.json md5.txt cylinder.zip
#    VITRA: hf_hub_download annotation/<id>.h5 + videos/<id>.mp4 into raw_data/vitra
#    RealDex: extract everything except images from the zip into raw_data/realdex/extracted

# 2. convert -> unified/   (optionally: a subset of dataset ids)
.venv/bin/python -m unidex.convert

# 3. coverage report -> unified/REPORT.md
.venv/bin/python -m unidex.report

# 4. tests (FK vs yourdfpy reference, palm frame, mirroring, mappings)
.venv/bin/python -m pytest -q tests
```

Visual checks: `unidex.viz.panel(dataset_id, episode_key, side, camera_id=...)` -> `unified/validation/*.png`.

## Where things are

| | |
|---|---|
| `config/hands.yaml` | hand model registry (URDF, palm/tip definitions, model status, validation log) |
| `config/datasets.yaml` | selected datasets, access, sample size, hand variants -> model ids |
| `unidex/mappings.py` | dataset-native hand vector -> model joints, with evidence per mapping |
| `unidex/kinematics/` | batched URDF FK (mimic joints), canonical palm frame |
| `unidex/adapters/` | one adapter per dataset -> `unidex.schema.Episode` |
| `UNIFIED_DATASET.md` | on-disk format |
