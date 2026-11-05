# Changelog

## Unreleased: run on RTX 40-series GPUs and on downloaded audio

Tag the code before these changes as `capstone-v1`. That version produced the capstone results and is one of the baselines.

### Fixed
- **Structure analysis on RTX 40-series GPUs.** The published All-In-One Cog image is CUDA 11.7 and crashes on these GPUs with `cuFFT error: CUFFT_INTERNAL_ERROR`. The pipeline now uses a local image, `allin1-cu118`, built by `scripts/build_allin1_image.sh` from `third_party/allin1/cog.yaml`. That image uses CUDA 11.8, NATTEN compiled with CUDA, pinned dependencies and model weights stored in the image. Set `MASHUP_ALLIN1_IMAGE` to use another image.
- **Beat positions were never saved.** All-In-One writes `beat_positions`, but `structure.py` copied `beatPositions`, so the field was always empty. Nothing downstream used it yet, so capstone outputs are unaffected.
- **Relative output folders broke structure analysis.** Cog runs inside the song folder, so a relative input path pointed at the wrong place. Paths are now absolute.
- **Stale analyser output.** `output*.json` files left by earlier runs are deleted before each run, so an old file can't be picked up by mistake.
- **YouTube downloads.** The forced `android` player client and the custom User-Agent are removed: they now lead to HTTP 403 errors. yt-dlp is pinned to 2026.10.26 with its default extras.
- **Batch script.** It now accepts the column names used in `data/mashup_pairs.csv` (`song1`, `song2`, `mashup`) as well as `*_url` and `*_id`. Previously it failed on that file.
- **`tests/test_cli_smoke.py`** checked for URLs it never passed in, so it always failed.

### Added
- Local audio files as inputs to `mashup.cli` (song 1, song 2 and `--human-mashup-url`). Supported types: `.wav`, `.webm`, `.opus`, `.m4a`, `.mp3`, `.flac` and `.ogg`. They are decoded to 44.1 kHz stereo WAV.
- `scripts/batch_evaluate.py`:
  - `--audio-dir` resolves video IDs and URLs to downloaded files.
  - `--start` and `--limit` run part of the CSV.
  - Rows marked `usable=0` are skipped.
  - If a pair fails, the error is recorded and the batch carries on.
  - Progress is saved after every pair, and a rerun resumes.
  - The output has `human_*` columns matching the `machine_*` ones.
- `tests/test_structure_and_inputs.py`: these tests need no Docker, GPU or network.

### Behaviour differences from `capstone-v1` to keep in mind when comparing results
- Local inputs are decoded to 44.1 kHz. Capstone downloads kept YouTube's sample rate, often 48 kHz, and fed that to HT-Demucs, which was trained at 44.1 kHz.
- The structure analyser now runs on the GPU. The capstone ran it on the CPU, because Docker had no GPU access. Expect small numerical differences in beats and segment boundaries.
- The mashup algorithm itself is unchanged.
