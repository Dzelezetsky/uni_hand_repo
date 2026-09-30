"""Plain-file metric logging for Stage-1 (works without wandb): averages every numeric entry of output_batch over
`every_n` training iterations and over the whole validation pass, prints it and appends it to
<job.path_local>/stage1_metrics.jsonl (rank 0). This file is what to send back from the server."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from imaginaire.utils import distributed, log
from imaginaire.utils.callback import Callback


class Stage1Metrics(Callback):
    def __init__(self, config=None, trainer=None, every_n: int = 50):
        self.every_n = every_n  # (mimic-video's CallBackGroup sets self.config / self.trainer after construction)
        self.train, self.val = defaultdict(list), defaultdict(list)
        self.t0 = time.time()

    @staticmethod
    def _add(acc, out):
        for k, v in out.items():
            if isinstance(v, (int, float)):
                acc[k].append(float(v))

    @distributed.rank0_only
    def _write(self, split, acc, iteration):
        row = {"split": split, "iter": iteration, "time_s": round(time.time() - self.t0, 1),
               **{k: sum(v) / len(v) for k, v in acc.items() if v}}
        log.info(f"[stage1 {split} {iteration}] " + " ".join(f"{k}={v:.4g}" for k, v in row.items()
                                                             if isinstance(v, float) and k != "time_s"))
        cfg = getattr(self, "config", None)
        if cfg is not None:
            path = Path(cfg.job.path_local) / "stage1_metrics.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0):
        self._add(self.train, output_batch)
        if iteration % self.every_n == 0:
            self._write("train", self.train, iteration)
            self.train = defaultdict(list)

    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0):
        self._add(self.val, output_batch)

    def on_validation_end(self, model, iteration: int = 0):
        self._write("val", self.val, iteration)
        self.val = defaultdict(list)
