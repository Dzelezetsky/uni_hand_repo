#!/usr/bin/env bash
# Wait for the matched-object downloads, then extract -> convert -> grasp moments -> grasp analysis.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while pgrep -f "realdex_sample.py" >/dev/null || pgrep -f "hrdexdb_objects.py" >/dev/null; do sleep 60; done
grep -h DONE raw_data/hrdexdb_objects.log raw_data/realdex_dl_?.log
$PY scripts/download/realdex_extract.py body_lotion sprayer blue_cup elephant_watering_can box goji_jar
$PY -m unidex.convert realdex hrdexdb | tail -n 3
$PY -m unidex.moments
$PY scripts/analysis/cluster_grasps.py 2>&1 | grep -v -i warn
$PY -m unidex.report > /dev/null
$PY -m pytest -q tests | tail -n 1
echo PIPELINE_DONE
