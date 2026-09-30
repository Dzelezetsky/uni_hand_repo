# RUNBOOK_SERVER.md — Stage-1 on the training server (2x A100 80 GB)

Every step below was run locally first (RTX 3060; the full model only in a truncated form). Run the commands
**from the repo root** unless a step says otherwise. After the steps marked **→ send**, paste the requested output
back into the chat.

Disk plan (~600 GB): video ~262 GB, hand/state data ~30 GB, unified + stage1_data ~55 GB, environments ~20 GB,
T5-11B 45 GB (temporary, deleted after step 9), training checkpoints ~80 GB (pruned, step 13).

---

## 0. Big-disk layout (once)

Put all large, git-ignored directories on the big disk and link them into the repo:

```bash
BIG=/path/to/big/disk/unidex          # <- EDIT
mkdir -p $BIG/{raw_data,unified_server,stage1_data,stage1_runs,mimic_checkpoints}
```

## 1. Code

```bash
git clone git@github.com:Dzelezetsky/uni_hand_repo.git
cd uni_hand_repo
git checkout stage1-mimic-video
ln -s $BIG/raw_data raw_data
ln -s $BIG/unified_server unified_server
ln -s $BIG/stage1_data stage1_data
ln -s $BIG/stage1_runs stage1_runs
ln -s $BIG/mimic_checkpoints third_party/mimic_video/model/checkpoints
```

Later updates: `git pull` (nothing in the tracked tree is modified on the server).

## 2. UnifiedDex environment (data pipeline)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # if uv is missing
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements-unidex.txt
.venv/bin/hf auth login                                  # account zelezetsky (gated: Origami, OpenArm Banana)
export UNIDEX_UNIFIED=$PWD/unified_server                # ALWAYS set this on the server (also in steps 5-6)
```

## 3. Hand models

```bash
bash stage1/fetch_hand_models.sh
```
Expected last line: `hand models ready`.

## 4. Datasets (hours; can run in the background)

```bash
.venv/bin/python stage1/download_data.py --dry-run       # prints sizes, expect TOTAL data+meta ~30 GB, video ~262 GB
nohup .venv/bin/python stage1/download_data.py > raw_data/download.log 2>&1 &
grep -E "done|TOTAL|Error" raw_data/download.log         # progress; re-run the same command after any network error
```

## 5. Conversion to the unified store + posture labels

```bash
export UNIDEX_UNIFIED=$PWD/unified_server
.venv/bin/python -m unidex.convert humanoid_everyday_h1 openarm_banana trex dexora egosteer sharpa_origami \
    > raw_data/convert.log 2>&1
tail -n 3 raw_data/convert.log
.venv/bin/python -m unidex.posture
```
Expected (verified locally): `unidex.posture` prints one line per dataset with these sample counts:
dexora 5,836,488 · egosteer 35,681,016 · humanoid_everyday_h1 1,923,845 · openarm_banana 234,978 ·
sharpa_origami 21,572,208 · trex 10,946,918 (±0; any difference means different data).
**→ send** the `unidex.posture` output.

## 6. Stage-1 index (video stays in raw_data)

```bash
export UNIDEX_UNIFIED=$PWD/unified_server
.venv/bin/python stage1/build_data.py humanoid_everyday_h1 openarm_banana trex dexora egosteer sharpa_origami
```
Expected: episodes written ≈ H1 4,882 · Banana 1,072 · T-Rex 5,464 · Dexora ~11,500 · EgoSteer ~13,600
(25 % of the video files) · Origami ~900 (50 % of the seasons); `skipped` = episodes without downloaded video.
**→ send** the output.

## 7. mimic-video environment

```bash
cd third_party/mimic_video/model
uv sync --extra cu126                                     # torch 2.6+cu126, flash-attn, transformer-engine, apex
cd ../../..
source stage1/env.sh                                     # ALWAYS before anything with $MIMIC_PY / torchrun
cd $MIMIC_MODEL && $MIMIC_PY scripts/test_environment.py && cd $UNIDEX_REPO
```
Expected: `Cosmos-predict2 environment setup is successful!`

## 8. Checkpoints (Cosmos-Predict2 2B, tokenizer, Bridge decoder, T5-11B)

```bash
source stage1/env.sh; cd $MIMIC_MODEL
$MIMIC_PY scripts/download_checkpoints.py --models pretrained_cosmos_bridge   # ~51 GB incl. T5-11B
cd $UNIDEX_REPO
```

## 9. T5 embeddings (one GPU), then delete T5

```bash
source stage1/env.sh; cd $MIMIC_MODEL
CUDA_VISIBLE_DEVICES=0 $MIMIC_PY $UNIDEX_REPO/stage1/precompute_t5.py
rm $MIMIC_MODEL/checkpoints/text_encoder/t5-11b/pytorch_model.bin           # frees 45 GB (not needed for training)
cd $UNIDEX_REPO
```

## 10. Dataset check

```bash
source stage1/env.sh
$MIMIC_PY stage1/scripts/check_dataset.py 3
```
Prints one line per dataset (video shape (3, 61, 480, 640), sample time) and writes `stage1_data/check_<dataset>.png`.
**→ send** the printed lines (and a PNG if something looks wrong).

## 11. Full-model smoke test: memory, speed, lambda

```bash
source stage1/env.sh; cd $MIMIC_MODEL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 $MIMIC_PY $UNIDEX_REPO/stage1/scripts/smoke_train.py --t5-subdir t5 --lora-rank 256 --steps 3 --batch 1
CUDA_VISIBLE_DEVICES=0 $MIMIC_PY $UNIDEX_REPO/stage1/scripts/smoke_train.py --t5-subdir t5 --lora-rank 256 --steps 3 --batch 2
CUDA_VISIBLE_DEVICES=0 $MIMIC_PY $UNIDEX_REPO/stage1/scripts/smoke_train.py --t5-subdir t5 --lora-rank 256 --steps 1 --grad-split 10
cd $UNIDEX_REPO
```
**→ send** the `step …`, `peak GPU memory …` and `median |grad| …` lines. From them we fix the per-GPU batch,
gradient accumulation and lambda (`model.config.hand_weight`).

## 12. Training

Template (the values of `hand_weight`, `batch_size`, `grad_accum_iter` come from step 11):
```bash
source stage1/env.sh; cd $MIMIC_MODEL
nohup torchrun --nproc_per_node=2 -m scripts.train --config=stage1_config.py -- experiment=stage1_clean \
    model.config.hand_weight=LAMBDA dataloader_train.batch_size=B trainer.grad_accum_iter=G \
    job.name=clean_v1 > $UNIDEX_REPO/stage1_runs/clean_v1.log 2>&1 &
```
Monitoring:
```bash
tail -n 3 $UNIDEX_REPO/stage1_runs/unidex_stage1/stage1/clean_v1/stage1_metrics.jsonl
nvidia-smi
```
A restarted command with the same `job.name` resumes from the last checkpoint.
Experiments: `stage1_clean` (main), `stage1_leaky` (ablation: hand loss at all video noise levels),
`stage1_off` (video only baseline; same data and steps).

## 13. Disk hygiene during training

```bash
.venv/bin/python stage1/prune_checkpoints.py stage1_runs/unidex_stage1/stage1/clean_v1 --keep 3 --dry-run
.venv/bin/python stage1/prune_checkpoints.py stage1_runs/unidex_stage1/stage1/clean_v1 --keep 3
```
Keeps the last 3 iterations and the best one by validation `hand/ce`.

## What to send back regularly

- the last lines of `stage1_metrics.jsonl` (train every 50 iterations, val every 1000);
- any traceback from the `.log` file (the lines after `Traceback`).
