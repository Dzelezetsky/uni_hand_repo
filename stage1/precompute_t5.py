"""T5-11B embeddings for every unique instruction of the Stage-1 index (mimic-video venv, one GPU).

  source stage1/env.sh
  $MIMIC_PY stage1/precompute_t5.py                       # server: checkpoints/text_encoder/t5-11b (45 GB, fp32)
  $MIMIC_PY stage1/precompute_t5.py --test-model google-t5/t5-large    # local smoke test ONLY (same 1024-d, other
                                                                         weights) -> stage1_data/t5_TESTONLY/
Same encoder call as mimic-video (CosmosT5TextEncoder.encode_prompts, max_length 512), but only the real tokens are
stored ([n_tokens, 1024] float32 under "encoded_text"); stage1/dataset.py zero-pads to 512 and builds the mask.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import safetensors.torch as st
import torch
import tqdm

REPO = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "stage1_data")
    ap.add_argument("--model-dir", default=str(Path(os.environ.get("MIMIC_MODEL", REPO / "third_party/mimic_video/model"))
                                              / "checkpoints/text_encoder/t5-11b"))
    ap.add_argument("--test-model", default=None, help="HF id of a small 1024-d T5 for a local pipeline test")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    instr = json.loads((a.root / "instructions.json").read_text())
    out = a.root / ("t5_TESTONLY" if a.test_model else "t5")
    out.mkdir(exist_ok=True)
    todo = [(k, v) for k, v in instr.items() if not (out / f"{k}.safetensors").exists()]
    print(f"{len(instr)} unique instructions, {len(todo)} to encode -> {out}")
    from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig
    enc = CosmosT5TextEncoder(config=CosmosT5TextEncoderConfig(ckpt_path=a.test_model or a.model_dir))
    with torch.inference_mode():
        for i in tqdm.tqdm(range(0, len(todo), a.batch)):
            chunk = todo[i:i + a.batch]
            emb, mask = enc.encode_prompts([v for _, v in chunk], max_length=512, return_mask=True)
            for (k, _), e, m in zip(chunk, emb, mask):
                st.save_file({"encoded_text": e[: int(m.sum())].float().contiguous().cpu()}, out / f"{k}.safetensors")
    print("done")
