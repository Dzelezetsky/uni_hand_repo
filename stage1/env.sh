# Source before running anything with the mimic-video venv:   source stage1/env.sh
# (works from any directory; sets MIMIC_MODEL / MIMIC_PY and fixes library lookup where no CUDA toolkit is installed)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNIDEX_REPO="$REPO"
export MIMIC_MODEL="$REPO/third_party/mimic_video/model"
export MIMIC_PY="$MIMIC_MODEL/.venv/bin/python"
SITE="$MIMIC_MODEL/.venv/lib/python3.10/site-packages"
# transformer-engine 1.13 looks for libnvrtc in CUDA_HOME / CUDA_PATH / /usr/local/cuda, then via `ldconfig -p`
# (which raises if nothing is found). Without a system CUDA toolkit, point it at the pip wheel nvidia-cuda-nvrtc-cu12.
if [ -z "${CUDA_HOME:-}${CUDA_PATH:-}" ] && ! ls /usr/local/cuda/lib64/libnvrtc.so* >/dev/null 2>&1 \
   && ! ldconfig -p 2>/dev/null | grep -q libnvrtc; then
  export CUDA_PATH="$SITE/nvidia/cuda_nvrtc"
fi
export PYTHONPATH="$MIMIC_MODEL:$REPO:${PYTHONPATH:-}"
# training outputs (checkpoints, stage1_metrics.jsonl) -> <repo>/stage1_runs/<project>/<group>/<name>
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$REPO/stage1_runs}"
# imaginaire/utils/distributed.py loads the UNVERSIONED "libcudart.so" (normally from a CUDA toolkit). Without one,
# expose the pip wheel's libcudart.so.12 under that name.
if ! ldconfig -p 2>/dev/null | grep -q "libcudart.so " && [ ! -e /usr/local/cuda/lib64/libcudart.so ]; then
  COMPAT="$MIMIC_MODEL/.venv/lib/cuda_compat"
  if [ ! -e "$COMPAT/libcudart.so" ] && [ -e "$SITE/nvidia/cuda_runtime/lib/libcudart.so.12" ]; then
    mkdir -p "$COMPAT" && ln -sf "$SITE/nvidia/cuda_runtime/lib/libcudart.so.12" "$COMPAT/libcudart.so"
  fi
  export LD_LIBRARY_PATH="$COMPAT:${LD_LIBRARY_PATH:-}"
fi
