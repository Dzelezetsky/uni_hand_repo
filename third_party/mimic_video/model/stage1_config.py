# UnifiedDex addition: entry point for `-m scripts.train --config=stage1_config.py` (scripts.train maps the path to a
# module relative to this directory). The actual config lives in <repo>/stage1/config.py (on PYTHONPATH via env.sh).
from stage1.config import make_config  # noqa: F401
