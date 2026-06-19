from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional

import soundfile as sf

from .assembly import assemble_full_mashup, fallback_vocals_over_mashup, time_stretch_audio
from .download import download_youtube_audio, ensure_dir
from .evaluation import evaluate_mashup
from .features import extract_all_stem_features_stretched, load_original_audio_into_stretched, precompute_all_stems
from .scoring import select_quartet
from .separation import extract_stems
from .structure import analyze_audio_and_save_json, load_song


def prepare_song_folder(base_folder, song_label, youtube_url):
    song_folder = ensure_dir(Path(base_folder) / song_label)
    original_wav = download_youtube_audio(youtube_url, song_folder, "original")
    extract_stems(original_wav, song_folder)
    json_filename = f"{song_label}.json"
    analyze_audio_and_save_json(original_wav, song_folder, json_filename)
    return str(song_folder), original_wav


def _candidate_bpms(song1_bpm: float, song2_bpm: float, target_bpm_strategy: str):
    if target_bpm_strategy.lower() == "min":
        return [min(song1_bpm, song2_bpm)]
    if target_bpm_strategy.lower() == "avg":
        return [0.5 * (song1_bpm + song2_bpm)]
    if target_bpm_strategy.lower() == "mid":
        return [0.5 * (song1_bpm + song2_bpm), song1_bpm, song2_bpm]
    try:
        return [float(target_bpm_strategy)]
    except Exception:
        return [0.5 * (song1_bpm + song2_bpm)]


def _select_best_bpm(songA, songB, candidate_bpms):
    best_bpm = candidate_bpms[0]
    best_score = -1e18
    for bpm_cand in candidate_bpms:
        rateA = bpm_cand / songA.bpm if songA.bpm > 0 else 1.0
        rateB = bpm_cand / songB.bpm if songB.bpm > 0 else 1.0
        score = -(abs(rateA - 1.0) + abs(rateB - 1.0))
        if score > best_score:
            best_score = score
            best_bpm = bpm_cand
    return best_bpm


def _apply_global_bpm(song, target_bpm, max_change_pct=0.06):
    from .features import read_wav, stretch_array_times, stretch_segments

    desired_rate = target_bpm / song.bpm if song.bpm > 0 else 1.0
    if abs(1.0 - desired_rate) > max_change_pct:
        desired_rate = 1.0 + (1.0 if desired_rate > 1 else -1.0) * max_change_pct

    stretched = {}
    for stem, p in song.audio_paths.items():
        y = read_wav(p, song.sr)
        y_st = time_stretch_audio(y, song.sr, rate=desired_rate)
        stretched[stem] = y_st

    song.stretched = stretched
    song.stretched_sr = song.sr
    song.stretched_beats = stretch_array_times(song.beats, desired_rate) if song.beats.size > 0 else song.beats
    song.stretched_downbeats = stretch_array_times(song.downbeats, desired_rate) if song.downbeats.size > 0 else song.downbeats
    song.stretched_segments = stretch_segments(song.segments, desired_rate)
    song.bpm = song.bpm * desired_rate
    return song


def main(
    song1_url: str,
    song2_url: str,
    output_dir: str,
    output_name: str,
    target_bpm_strategy: str = "mid",
    lam: float = 0.6,
    fast_mode: bool = True,
    parallel: bool = True,
    tsm_backend: str = "auto",
    human_mashup_url: Optional[str] = None,
    return_results: bool = False,
):
    start_time = time.time()
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    song1_folder, audio1_path = prepare_song_folder(output_dir, "song1", song1_url)
    song2_folder, audio2_path = prepare_song_folder(output_dir, "song2", song2_url)

    songA = load_song(song1_folder)
    songB = load_song(song2_folder)

    songA = load_original_audio_into_stretched(songA)
    songB = load_original_audio_into_stretched(songB)

    candidate_bpms = _candidate_bpms(songA.bpm, songB.bpm, target_bpm_strategy)
    print(f"Evaluating candidate BPMs: {candidate_bpms}")

    pcsA = precompute_all_stems(songA, fast_mode=fast_mode, parallel=False)
    pcsB = precompute_all_stems(songB, fast_mode=fast_mode, parallel=False)
    featsA = extract_all_stem_features_stretched(songA, pcsA)
    featsB = extract_all_stem_features_stretched(songB, pcsB)

    best_bpm = _select_best_bpm(songA, songB, candidate_bpms)
    print(f"Selected BPM: {best_bpm:.2f}")

    songA = _apply_global_bpm(songA, best_bpm, max_change_pct=0.06)
    songB = _apply_global_bpm(songB, best_bpm, max_change_pct=0.06)

    pcsA = precompute_all_stems(songA, fast_mode=fast_mode, parallel=False)
    pcsB = precompute_all_stems(songB, fast_mode=fast_mode, parallel=False)
    featsA = extract_all_stem_features_stretched(songA, pcsA)
    featsB = extract_all_stem_features_stretched(songB, pcsB)

    quartet, score = select_quartet(songA, songB, featsA, featsB, lam=lam)
    if quartet is not None:
        print("Diagnostic quartet selected.")
    else:
        print("Diagnostic quartet selection failed; proceeding with per-segment assembly.")

    y_mix = assemble_full_mashup(songA, songB, pcsA, pcsB, featsA, featsB, lam)
    output_path = str(output_dir / output_name)

    if y_mix.size == 0:
        print("Warning: Generated mashup is empty. Using fallback vocals-over instrumental mashup.")
        fallback_vocals_over_mashup(song1_folder, song2_folder, output_path)
    else:
        sf.write(output_path, y_mix, songA.stretched_sr)
        print(f"Wrote mashup to {output_path}")

    elapsed_seconds = time.time() - start_time
    print(f"Total processing time: {elapsed_seconds:.2f} seconds")

    json1_path = str(Path(song1_folder) / "song1.json")
    json2_path = str(Path(song2_folder) / "song2.json")
    machine_eval = evaluate_mashup(json1_path, json2_path, audio1_path, audio2_path, output_path)
    print("Machine evaluation:", machine_eval)

    human_eval = None
    if human_mashup_url:
        print(f"[INFO] Processing human mashup from URL: {human_mashup_url}")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            human_folder, human_wav = prepare_song_folder(tmpdir, "human_mashup", human_mashup_url)
            human_eval = evaluate_mashup(json1_path, json2_path, audio1_path, audio2_path, human_wav)
            print("Human evaluation:", human_eval)

    results = {
        "machine_eval": machine_eval,
        "human_eval": human_eval,
        "processing_time": elapsed_seconds,
        "output_path": output_path,
    }
    return results if return_results else None


def build_parser():
    parser = argparse.ArgumentParser(description="Multi-segment stem-level mashup with policy-driven per-segment selection")
    parser.add_argument("song1_url", type=str)
    parser.add_argument("song2_url", type=str)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--output-name", type=str, default="mashup.wav")
    parser.add_argument("--target-bpm", type=str, default="mid", help='Target BPM: "mid", "min", "avg", or numeric')
    parser.add_argument("--lambda_vh", type=float, default=0.6, help="Lambda for vertical vs horizontal stem score [0,1]")
    parser.add_argument("--no-parallel", action="store_true", help="Disable multiprocessing for precompute")
    parser.add_argument("--tsm-backend", type=str, default="auto", choices=["auto", "rubberband", "librosa"])
    parser.add_argument("--slow-mode", action="store_true", help="Disable fast-mode tweaks")
    parser.add_argument("--human-mashup-url", type=str, default=None, help="Optional YouTube URL for human mashup audio")
    return parser


def cli_main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    parser = build_parser()
    args = parser.parse_args()
    main(
        args.song1_url,
        args.song2_url,
        args.output_dir,
        args.output_name,
        target_bpm_strategy=args.target_bpm,
        lam=args.lambda_vh,
        fast_mode=not args.slow_mode,
        parallel=not args.no_parallel,
        tsm_backend=args.tsm_backend,
        human_mashup_url=args.human_mashup_url,
    )


if __name__ == "__main__":
    cli_main()
