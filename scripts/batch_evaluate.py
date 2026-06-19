from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mashup.cli import main


def run_batch(input_csv: str, output_csv: str, results_dir: str):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    rows = []
    for i, row in df.iterrows():
        song1_url = row["song1_url"]
        song2_url = row["song2_url"]
        human_mashup_url = row.get("human_mashup_url", None)
        output_name = row.get("output_name", f"mashup_{i}.wav")
        run_dir = results_dir / f"run_{i:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = main(
            song1_url,
            song2_url,
            output_dir=str(run_dir),
            output_name=output_name,
            human_mashup_url=human_mashup_url if isinstance(human_mashup_url, str) and human_mashup_url.strip() else None,
            return_results=True,
        )

        rows.append({
            "row_index": i,
            "song1_url": song1_url,
            "song2_url": song2_url,
            "human_mashup_url": human_mashup_url if isinstance(human_mashup_url, str) else None,
            "output_path": result["output_path"],
            "processing_time": result["processing_time"],
            "machine_final_score": result["machine_eval"]["final_score"],
            "machine_chroma_harmonic_similarity": result["machine_eval"]["chroma_harmonic_similarity"],
            "machine_clap_embedding_similarity": result["machine_eval"]["clap_embedding_similarity"],
            "machine_final_snr": result["machine_eval"]["final_snr"],
            "machine_peak": result["machine_eval"]["peak"],
            "human_final_score": None if result["human_eval"] is None else result["human_eval"]["final_score"],
        })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def build_parser():
    p = argparse.ArgumentParser(description="Batch evaluate mashup pairs from a CSV")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--results-dir", default="results")
    return p


def main_cli():
    args = build_parser().parse_args()
    run_batch(args.input_csv, args.output_csv, args.results_dir)


if __name__ == "__main__":
    main_cli()
