# Vendored copy of mimic-video

- Source: https://github.com/mimic-video/mimic-video
- Upstream commit: e3355dbc93132b576c02f920a59b4fc18a4f5906 (2026-06-25)
- License: Apache-2.0 (LICENSE here; model/LICENSE and model/ATTRIBUTIONS.md keep the Cosmos-Predict2 notices)
- Copied: `model/`, `data_preprocessing/`, LICENSE, README.md, MODEL.md, DATA.md.
  Not copied: `eval/` (LIBERO / SimplerEnv simulators and assets, 450 MB), `assets/` (README figures).
- The first commit that adds this directory contains the files UNMODIFIED. Every later change is ours
  (UnifiedDex Stage-1: hand head, joint loss, data loading); see `git log -- third_party/mimic_video` and
  `git diff <first commit> -- third_party/mimic_video`. Modified files are marked with a
  `# UnifiedDex modification` comment near the change, as required by Apache-2.0 §4(b).
