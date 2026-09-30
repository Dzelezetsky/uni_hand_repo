"""OpenArm Banana 1072 (June777/openarm_banana_all_1072episodes, gated, ~3.8 GB): meta, data and all four views."""
from huggingface_hub import snapshot_download

snapshot_download("June777/openarm_banana_all_1072episodes", repo_type="dataset", local_dir="raw_data/openarm_banana/all_1072",
                  max_workers=8)
print("DONE")
