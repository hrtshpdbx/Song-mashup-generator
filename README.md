# Automatic Music Mashup Generation

**Perceptually-guided automatic mashup creation through multi-feature compatibility scoring.**

Give the pipeline two songs and it builds a mashup of them: it separates each song into stems, finds the song structure, scores which parts of the two songs fit together, and assembles the result beat by beat. It can also score its own output against a human-made mashup of the same two songs, which is how the system was evaluated.

Capstone project, B.Tech in Artificial Intelligence, NITK Surathkal (2024–2025).

---

## Why this project

Mashups are made by people, one at a time. Music information retrieval has plenty of song datasets and a large literature on generating music, but very little on two-song mashups, and no public benchmark that shows what a good one should sound like. So this project had two parts: a system that makes a mashup from any two songs, and a benchmark to measure it against human work.

---

## How it works

```text
two YouTube URLs
      │
      ▼
download (yt-dlp) ──► WAV
      │
      ▼
4-stem separation (Demucs): vocals · drums · bass · other
      │
      ▼
structure analysis (All-In-One) + beat tracking
      │
      ▼
feature extraction: chroma · MFCC · RMS · tempo curves · pYIN pitch · CLAP embeddings
      │
      ▼
compatibility scoring on two axes
   vertical   – does this stem sound right on top of that one?  (timbre, harmony)
   horizontal – does this section follow that one?              (structure, time)
      │
      ▼
segment selection over a canonical 13-segment song structure
      │
      ▼
key-aware assembly · beat-synchronous time-stretching (Rubber Band) · 20 ms crossfades
      │
      ▼
loudness balancing · mastering (pedalboard) · export
```

A single mashup takes about 20 minutes on CPU. A CUDA GPU speeds up stem separation considerably.

---

## Evaluation and results

### The benchmark

No public dataset pairs two source songs with a human-made mashup of exactly those two songs, so I built one. Each entry is a triple found on YouTube: song 1, song 2, and a human mashup made from only those two songs. Most human mashups use more than two songs, and those were excluded.

| | Pairs |
|---|---|
| Collected | 450 |
| Evaluated | **408** |
| Dropped | 42, because the human mashup had been deleted or taken down for copyright between collection and evaluation |

### The measures

The generated mashups and the human mashups are scored on the same four measures, with the same code:

| Measure | What it captures |
|---|---|
| Harmonic coherence | chroma-based harmonic similarity |
| Semantic similarity | CLAP embedding similarity |
| SNR | signal-to-noise ratio of the output |
| Technical fidelity | a technical audio-quality score for the output |

### Results

| | Mean score (408 pairs) |
|---|---|
| Human-made mashups | **0.371959** |
| Generated mashups | **0.361459** |

The two are close on harmony and on audio quality. The gap sits mostly in arrangement-level judgement, such as which song's bass belongs under the other song's chorus. That is the part of mashup-making that objective measures capture least.

**Known limitations:** transient preservation, harmonic clashes on some pairs, and arrangement choices that a human producer would make differently.

**One lesson worth recording.** Stem levels were wrong for a long time because they were being balanced by RMS. RMS measures the average energy in a signal, not how loud it sounds to a listener. Weeks of equalisation and crossfade tuning were spent fixing a mix that was not the problem, when the fault was in the measurement. Switching to a perceptual loudness measure fixed it.

**Planned:** a listening study comparing outputs from different versions of the pipeline, to cover the subjective part that the four measures miss.

---

## Repository structure

```text
Song-mashup-generator/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── setup.sh             # one-step setup for Ubuntu
├── configs/
├── data/                # input CSVs for batch evaluation
├── scripts/             # batch_evaluate.py
├── src/                 # the `mashup` package (CLI: python -m mashup.cli)
└── tests/
```

---

## Installation

### Requirements

- Python 3.10 or later
- FFmpeg
- Rubber Band CLI, for high-quality time-stretching
- Docker and Cog, for the All-In-One structure analyser
- Optional: an NVIDIA GPU with CUDA. Everything also runs on CPU.

Tested on Ubuntu 24.04 LTS.

### 1. Clone this repository and the structure analyser side by side

The pipeline calls the All-In-One Music Structure Analyzer from a folder named `cog-all-in-one` that sits **next to** this repository:

```text
projects/
├── Song-mashup-generator/
└── cog-all-in-one/
```

```bash
mkdir projects && cd projects
git clone https://github.com/hrtshpdbx/Song-mashup-generator.git
git clone https://github.com/sakemin/all-in-one.git cog-all-in-one
```

### 2. Set up the structure analyser

```bash
cd cog-all-in-one
```

Follow **all** of the installation instructions in the analyser's repository: https://github.com/sakemin/all-in-one

Before going further, check that Cog and Docker are installed, that the models and checkpoints have downloaded, and that the repository's own example commands run.

Then go back:

```bash
cd ../Song-mashup-generator
```

### 3. Install the system tools

| | Ubuntu | macOS | Windows |
|---|---|---|---|
| **Docker** | `sudo apt install -y docker.io` then `sudo systemctl enable --now docker` | [Docker Desktop](https://www.docker.com/products/docker-desktop/) | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| **FFmpeg** | `sudo apt install -y ffmpeg` | `brew install ffmpeg` | `choco install ffmpeg`, or from [ffmpeg.org](https://ffmpeg.org/download.html) |
| **Rubber Band** | `sudo apt install -y rubberband-cli` | `brew install rubberband` | `choco install rubberband`, or install manually and add it to `PATH` |

On macOS and Windows, launch Docker Desktop and wait until it reports that it is running.

Check all three:

```bash
docker --version && docker ps
ffmpeg -version
rubberband --help
```

### 4. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**On Ubuntu, `setup.sh` does steps 3 (FFmpeg and Rubber Band only) and 4 in one go:**

```bash
chmod +x setup.sh
./setup.sh
```

It uses `apt`, so it is Ubuntu-only. On macOS and Windows, follow steps 3 and 4 by hand. Docker and the structure analyser are not installed by the script.

### 5. Check the installation

```bash
python -c "import mashup; print('Installation successful')"
```

**First run.** The first run is slow. Demucs downloads its pretrained models, All-In-One may download its own, PyTorch initialises CUDA, and cache directories are created. Later runs are much faster.

---

## Usage

### Make one mashup

```bash
python -m mashup.cli \
  "https://www.youtube.com/watch?v=VIDEO_ID_1" \
  "https://www.youtube.com/watch?v=VIDEO_ID_2" \
  --output-dir outputs/run1 \
  --output-name mashup.wav
```

### Batch evaluation

Batch evaluation makes a mashup for each song pair in a CSV and scores it. If you give a human mashup URL for a pair, it scores that too.

**1. Prepare the input CSV** at `data/mashup_pairs.csv`:

```csv
song1_url,song2_url,human_mashup_url
https://www.youtube.com/watch?v=...,https://www.youtube.com/watch?v=...,https://www.youtube.com/watch?v=...
```

The `human_mashup_url` column is optional.

**2. Run it:**

```bash
python scripts/batch_evaluate.py \
  --input-csv data/mashup_pairs.csv \
  --output-csv outputs/mashup_eval.csv \
  --results-dir outputs/batch_results
```

**3. The outputs:**

```text
outputs/
├── batch_results/
│   ├── pair_000/
│   ├── pair_001/
│   └── ...
└── mashup_eval.csv      # paths to the generated mashups, and the evaluation scores
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: ffmpeg` | Install FFmpeg and make sure it is on your `PATH`. |
| `FileNotFoundError: rubberband` | Install the Rubber Band CLI and make sure it is on your `PATH`. |
| `Cannot connect to the Docker daemon` | Start Docker Desktop (macOS or Windows) or the Docker service (Linux), then check with `docker ps`. |
| Structure analysis fails | Check that `cog-all-in-one` sits next to this repository, that Docker is running, that Cog is installed, that all checkpoints have downloaded, and that the analyser's own example commands work. |
| First run seems stuck | Demucs is downloading its models. Let it finish. |
| YouTube download fails | Run `pip install -U yt-dlp`. |
| CUDA not used | `nvidia-smi` should list your GPU and `python -c "import torch; print(torch.cuda.is_available())"` should print `True`. If not, reinstall PyTorch with the CUDA build for your platform. |

---

## Reproducibility

Downloaded audio, generated mashups, caches, virtual environments, temporary evaluation outputs and model downloads are kept out of version control. They are all recreated when the pipeline runs.

## Responsible use

The pipeline downloads audio from YouTube to process it. Use it for research and personal experimentation, respect the rights of the original artists and uploaders, and do not redistribute downloaded audio or mashups built from copyrighted recordings. No audio is included in this repository.

---

## Credits and licence

Built with [yt-dlp](https://github.com/yt-dlp/yt-dlp), [Demucs](https://github.com/facebookresearch/demucs), the [All-In-One Music Structure Analyzer](https://github.com/sakemin/all-in-one) (Cog packaging), [librosa](https://librosa.org/), [torchaudio](https://pytorch.org/audio/), CLAP audio embeddings, [Rubber Band](https://breakfastquay.com/rubberband/), [pedalboard](https://github.com/spotify/pedalboard), [pydub](https://github.com/jiaaro/pydub) and [SciPy](https://scipy.org/). Each of these is used under its own licence.

The code in this repository is released under the **MIT License**; see [`LICENSE`](LICENSE).

**Author:** Smruthi Bhat
