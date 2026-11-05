"""Run the mashup pipeline over many (song1, song2, human mashup) rows and collect the metrics.

Input CSV: one row per pair. Column names accepted for each role (first match wins):
    song 1:        song1_url, song1, song1_id
    song 2:        song2_url, song2, song2_id
    human mashup:  human_mashup_url, mashup, mashup_id      (optional)
Values can be YouTube URLs, local audio paths, or YouTube video IDs. With --audio-dir, URLs and IDs
are resolved to files saved by tools/fetch_audio.py (<audio-dir>/<video_id>.<ext>), so nothing is
downloaded again. A `usable` column (as in link_check/triples.csv), if present, skips rows marked 0.

The output CSV is rewritten after every pair, and rows already marked "ok" are skipped on the next
run, so an interrupted batch can be resumed with the same command. A failing pair is recorded with
status "error" and the batch carries on.

    python scripts/batch_evaluate.py --input-csv link_check/triples.csv --audio-dir ~/mashup_audio \
        --output-csv results/baseline_v1.csv --results-dir results/baseline_v1 --limit 3
"""
from __future__ import annotations

import argparse
import re
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

ROLE_COLUMNS = {
    "song1": ["song1_url", "song1", "song1_id"],
    "song2": ["song2_url", "song2", "song2_id"],
    "mashup": ["human_mashup_url", "mashup", "mashup_id"],
}
METRICS = ["final_score", "chroma_harmonic_similarity", "clap_embedding_similarity",
           "technical_peak_and_snr_score", "final_snr", "peak"]
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_ID_RE = re.compile(r"(?:youtu\.be/|[?&]v=|/shorts/)([A-Za-z0-9_-]{11})")


def resolve_columns(columns) -> dict:
    cols = {}
    for role, candidates in ROLE_COLUMNS.items():
        cols[role] = next((c for c in candidates if c in columns), None)
    missing = [r for r in ("song1", "song2") if cols[r] is None]
    if missing:
        raise ValueError(f"Input CSV has no column for {missing}; accepted names: "
                         + "; ".join(f"{r}: {ROLE_COLUMNS[r]}" for r in missing))
    return cols


def video_id(value: str) -> Optional[str]:
    value = value.strip()
    if _ID_RE.match(value):
        return value
    m = _URL_ID_RE.search(value)
    return m.group(1) if m else None


def resolve_source(value, audio_dir: Optional[Path]) -> Optional[str]:
    """Return a local file path or URL for the pipeline, or None if the cell is empty."""
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if Path(value).expanduser().is_file():
        return str(Path(value).expanduser())
    vid = video_id(value)
    if audio_dir is not None and vid is not None:
        matches = sorted(p for p in audio_dir.glob(f"{vid}.*") if p.suffix not in {".part", ".json", ".csv"})
        if matches:
            return str(matches[0])
        raise FileNotFoundError(f"No downloaded audio for video {vid} in {audio_dir}")
    if vid is not None and not value.startswith("http"):
        return f"https://www.youtube.com/watch?v={vid}"
    return value


def _flatten(prefix: str, ev: Optional[dict]) -> dict:
    return {f"{prefix}_{m}": (None if ev is None else ev.get(m)) for m in METRICS}


def run_batch(input_csv, output_csv, results_dir, audio_dir=None, start=0, limit=None, include_unusable=False):
    from mashup.cli import main  # imported here so --help works without the heavy dependencies

    output_csv, results_dir = Path(output_csv), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(audio_dir).expanduser() if audio_dir else None

    df = pd.read_csv(input_csv, dtype=str)
    cols = resolve_columns(df.columns)
    rows = df.iloc[start:] if limit is None else df.iloc[start:start + limit]

    done = {}
    if output_csv.exists():
        prev = pd.read_csv(output_csv)
        done = {int(r["row_index"]): r.to_dict() for _, r in prev.iterrows()}
    results = dict(done)

    for i, row in rows.iterrows():
        if done.get(i, {}).get("status") == "ok":
            continue
        if not include_unusable and "usable" in df.columns and str(row.get("usable", "1")).strip() in {"0", "False", "false"}:
            continue
        record = {"row_index": i, "song1": row[cols["song1"]], "song2": row[cols["song2"]],
                  "mashup": row[cols["mashup"]] if cols["mashup"] else None}
        t0 = time.time()
        try:
            s1 = resolve_source(row[cols["song1"]], audio_dir)
            s2 = resolve_source(row[cols["song2"]], audio_dir)
            hm = resolve_source(row[cols["mashup"]], audio_dir) if cols["mashup"] else None
            run_dir = results_dir / f"run_{i:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = main(s1, s2, output_dir=str(run_dir), output_name="mashup.wav",
                          human_mashup_url=hm, return_results=True)
            record.update({"status": "ok", "error": None, "output_path": result["output_path"],
                           "processing_time": result["processing_time"]})
            record.update(_flatten("machine", result["machine_eval"]))
            record.update(_flatten("human", result["human_eval"]))
        except Exception as exc:  # keep going; the error is recorded for this row
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:500],
                           "processing_time": time.time() - t0})
            traceback.print_exc()
        results[i] = record
        pd.DataFrame([results[k] for k in sorted(results)]).to_csv(output_csv, index=False)
        print(f"[{i}] {record['status']} in {record['processing_time']:.0f} s")

    return output_csv


def build_parser():
    p = argparse.ArgumentParser(description="Batch-run and evaluate mashup pairs from a CSV")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--audio-dir", default=None, help="folder with <video_id>.<ext> files from tools/fetch_audio.py")
    p.add_argument("--start", type=int, default=0, help="first row (0-based) to process")
    p.add_argument("--limit", type=int, default=None, help="process at most this many rows")
    p.add_argument("--include-unusable", action="store_true", help="don't skip rows with usable=0")
    return p


def main_cli():
    args = build_parser().parse_args()
    run_batch(args.input_csv, args.output_csv, args.results_dir, audio_dir=args.audio_dir,
              start=args.start, limit=args.limit, include_unusable=args.include_unusable)


if __name__ == "__main__":
    main_cli()
