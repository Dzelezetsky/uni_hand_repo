#!/bin/bash
# Fetch the hand models used by the Stage-1 datasets into external_hand_models/ (git-ignored) at the commits used for
# verification, and rebuild the derived URDFs. Run from the repo root with the UnifiedDex venv:  bash stage1/fetch_hand_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=external_hand_models
mkdir -p $M

clone () {  # url dir commit
  if [ ! -d "$2/.git" ]; then git clone --quiet "$1" "$2"; fi
  git -C "$2" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "$2" checkout --quiet "$3"
  echo "$2 @ $(git -C "$2" rev-parse --short HEAD)"
}
clone https://github.com/unitreerobotics/xr_teleoperate.git        $M/unitree_xr_teleoperate 817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b
clone https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git   $M/sharpa-urdf-usd-xml    0d19cac602f46456b819e4b6a2c09a74982c9a3e
clone https://github.com/roboterax/models.git                      $M/roboterax_models/models e8660e664e39e5b80ebb1f252e1b77b0e6929cff
$PY scripts/models/extract_star1_xhand.py                     # -> roboterax_models/xhand1_{right,left}_star1.urdf

# EgoSteer RY-H2: authors' MJCF + FK node from robot-stack, converted to URDF
TMP=$(mktemp -d)
git clone --quiet https://github.com/egosteer/robot-stack "$TMP/robot-stack"
git -C "$TMP/robot-stack" checkout --quiet ba06f62efdc66cd7d8f52d1475147fc5c84fe27c
mkdir -p $M/egosteer_ruiyan
cp -r "$TMP/robot-stack/assets/ruiyan_hand_mjcf" $M/egosteer_ruiyan/
cp "$TMP/robot-stack/src/hand/hand/hand_fk_node.py" $M/egosteer_ruiyan/
echo "egosteer/robot-stack ba06f62efdc66cd7d8f52d1475147fc5c84fe27c (Apache-2.0 per dataset card)" > $M/egosteer_ruiyan/SOURCE.txt
rm -rf "$TMP"
$PY scripts/models/mjcf_to_urdf_ruiyan.py                     # -> egosteer_ruiyan/ruiyan_hand_mjcf/*/ruiyan_ryh2_*.urdf

# Inspire RH56F1 (HRDexDB authors' URDF + meshes; also used for OpenArm Banana)
$PY - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
# list only assets/robots: snapshot_download(allow_patterns=...) walks the whole (huge) repo tree first and hangs
files = [f.path for f in HfApi().list_repo_tree("HRDexDB/HRDexDB", repo_type="dataset", path_in_repo="assets/robots",
                                                recursive=True) if hasattr(f, "size")]
for p in files:
    hf_hub_download("HRDexDB/HRDexDB", p, repo_type="dataset", local_dir="raw_data/hrdexdb")
print(f"raw_data/hrdexdb/assets/robots ok ({len(files)} files)")
PY
echo "hand models ready"
