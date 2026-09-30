"""EgoSteer-RealWorld: all frame-level parquet (state/action/extrinsics, ~17 GB) + meta, no video."""
from huggingface_hub import snapshot_download

snapshot_download("EgoSteer/EgoSteer-RealWorld", repo_type="dataset", allow_patterns=["meta/*", "meta/episodes/*/*.parquet",
                  "data/*/*.parquet"], local_dir="raw_data/egosteer", max_workers=8)
print("DONE")
