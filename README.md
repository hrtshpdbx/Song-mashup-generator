# Automatic Music Mashup Generation

Automatic music mashup generation from two YouTube tracks using stem separation, structure analysis, beat alignment, and compatibility scoring. The pipeline downloads each source track, separates stems with Demucs, analyzes song structure, selects compatible segments across both songs, assembles a full mashup, and optionally compares the machine-generated result against a human mashup reference.

## Features

- YouTube input
- Demucs stem separation
- structure-aware segment selection
- optional human-vs-machine evaluation

## Repository structure

```text
capstone-mashup/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── setup.sh
├── configs/
├── data/
├── scripts/
├── src/
└── tests/
```
## Installation

### System Requirements

Tested on:

* Ubuntu 24.04 LTS
* Python 3.10+
* FFmpeg
* Rubber Band CLI
* Docker
* NVIDIA GPU (recommended, optional)

A CUDA-capable GPU is recommended for faster stem separation, but the pipeline can also run entirely on CPU.

---

### Clone the Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/capstone-mashup.git
cd capstone-mashup
```

---

## Install All-In-One Next To This Repository

This project uses the All-In-One Music Structure Analyzer for song structure detection.

The mashup pipeline expects the analyzer repository to exist locally in a folder named:

```text
cog-all-in-one
```

Clone both repositories into the same parent directory:

```text
projects/
├── capstone-mashup/
└── cog-all-in-one/
```

Example:

```bash
mkdir projects
cd projects

git clone https://github.com/<YOUR_USERNAME>/capstone-mashup.git
git clone https://github.com/sakemin/all-in-one.git cog-all-in-one
```

After cloning, your directory structure should look like:

```text
projects/
├── capstone-mashup
└── cog-all-in-one
```

The mashup code calls the structure analyzer from this local folder.

---

### Install the All-In-One Analyzer

Enter the analyzer repository:

```bash
cd ../cog-all-in-one
```

Follow **all installation instructions** provided in the All-In-One repository:

https://github.com/sakemin/all-in-one

Make sure the following work successfully before returning to this repository:

* Cog installation
* Docker installation
* Model downloads
* Checkpoint downloads
* Example structure-analysis commands

Once the analyzer is working, return to the mashup repository:

```bash
cd ../capstone-mashup
```

---

## Install Docker

The All-In-One analyzer uses Cog and Docker.

### Ubuntu

```bash
sudo apt update
sudo apt install -y docker.io
```

Enable Docker:

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

Verify:

```bash
docker --version
docker ps
```

---

### Windows

Install Docker Desktop:

https://www.docker.com/products/docker-desktop/

After installation:

1. Launch Docker Desktop
2. Wait until Docker reports "Running"

Verify:

```powershell
docker --version
docker ps
```

---

### macOS

Install Docker Desktop:

https://www.docker.com/products/docker-desktop/

Launch Docker Desktop and wait until Docker is running.

Verify:

```bash
docker --version
docker ps
```

---

## Install FFmpeg

### Ubuntu

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Verify:

```bash
ffmpeg -version
```

---

### Windows

Install FFmpeg using Chocolatey:

```powershell
choco install ffmpeg
```

Or download from:

https://ffmpeg.org/download.html

Verify:

```powershell
ffmpeg -version
```

---

### macOS

Install using Homebrew:

```bash
brew install ffmpeg
```

Verify:

```bash
ffmpeg -version
```

---

## Install Rubber Band CLI

Rubber Band is used for high-quality time stretching.

### Ubuntu

```bash
sudo apt install -y rubberband-cli
```

Verify:

```bash
rubberband --help
```

---

### Windows

Install using Chocolatey:

```powershell
choco install rubberband
```

If Chocolatey does not provide a working package, install Rubber Band manually and ensure the executable is available in your system PATH.

Verify:

```powershell
rubberband --help
```

---

### macOS

Install using Homebrew:

```bash
brew install rubberband
```

Verify:

```bash
rubberband --help
```

---

## Create the Python Environment

### Ubuntu / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## Install Python Dependencies

Upgrade pip:

```bash
pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package:

```bash
pip install -e .
```

---

## Using the Setup Script (Linux/macOS)

A convenience setup script is provided:

```bash
chmod +x setup.sh
./setup.sh
```

The script performs:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

sudo apt install -y ffmpeg
sudo apt install -y rubberband-cli

pip install -e .
```

Windows users should follow the manual installation steps above.

---

## Verify Installation

Activate the virtual environment.

Ubuntu/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Verify package import:

```bash
python -c "import mashup; print('Installation successful')"
```

Expected output:

```text
Installation successful
```

---

## First Run Notes

The first execution may take significantly longer because:

* Demucs downloads pretrained stem-separation models
* All-In-One may download required models
* PyTorch initializes CUDA resources
* Cache directories are created

Subsequent runs are much faster.

---

## Creating a Mashup

The pipeline automatically:

1. Downloads both YouTube tracks
2. Converts audio to WAV
3. Separates stems using Demucs
4. Detects song structure using All-In-One
5. Extracts musical features
6. Computes compatibility scores
7. Selects matching segments
8. Builds a mashup
9. Applies mastering and exports audio

### Run a Single Mashup

```bash
python -m mashup.cli \
  "https://www.youtube.com/watch?v=VIDEO_1" \
  "https://www.youtube.com/watch?v=VIDEO_2" \
  --output-dir outputs/run1 \
  --output-name mashup.wav
```

Example:

```bash
python -m mashup.cli \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  "https://www.youtube.com/watch?v=3JWTaaS7LdU" \
  --output-dir outputs/example \
  --output-name mashup.wav
```

---

## Batch Evaluation

Batch evaluation generates mashups for multiple song pairs and computes evaluation metrics.

### Prepare Input CSV

```csv
song1_url,song2_url,human_mashup_url
https://youtube.com/...,https://youtube.com/...,https://youtube.com/...
https://youtube.com/...,https://youtube.com/...,https://youtube.com/...
```

The `human_mashup_url` column is optional.

Save as:

```text
data/mashup_pairs.csv
```

---

### Run Batch Evaluation

```bash
python scripts/batch_evaluate.py \
  --input-csv data/mashup_pairs.csv \
  --output-csv outputs/mashup_eval.csv \
  --results-dir outputs/batch_results
```

---

### Batch Evaluation Outputs

```text
outputs/
├── batch_results/
│   ├── pair_000/
│   ├── pair_001/
│   └── ...
└── mashup_eval.csv
```

The evaluation CSV contains generated mashup paths and evaluation metrics.

---

## Troubleshooting

### FFmpeg Not Found

```text
FileNotFoundError: ffmpeg
```

Ensure FFmpeg is installed and available in your system PATH.

---

### Rubber Band Not Found

```text
FileNotFoundError: rubberband
```

Ensure Rubber Band CLI is installed and available in your system PATH.

---

### Docker Not Running

```text
Cannot connect to the Docker daemon
```

Start Docker Desktop (Windows/macOS) or Docker service (Linux).

Verify:

```bash
docker ps
```

---

### Structure Analysis Fails

Verify:

* `cog-all-in-one` exists next to this repository
* Docker is running
* Cog is installed
* All required checkpoints are downloaded
* Example commands from the All-In-One repository work successfully

Repository:

https://github.com/sakemin/all-in-one

---

### Demucs Download Delay

The first run downloads pretrained Demucs models automatically.

Wait for the download to finish before terminating the process.

---

### YouTube Download Failure

Update yt-dlp:

```bash
pip install -U yt-dlp
```

---

### CUDA Not Detected

Check:

```bash
nvidia-smi
```

and

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

If `False`, reinstall PyTorch with CUDA support appropriate for your platform.

---

## Reproducibility Notes

The following are intentionally excluded from version control:

* downloaded audio
* generated mashups
* cache directories
* virtual environments
* temporary evaluation outputs
* model downloads

All of these artifacts are recreated automatically during execution.

# System dependencies

- `ffmpeg`
- `cog` for structure analysis
- `rubberband` for the higher-quality time-stretch path, optional

Demucs will download models on first use.

# How to run one mashup

```bash
python -m mashup.cli   "YOUTUBE_URL_1"   "YOUTUBE_URL_2"   --output-dir outputs/run1   --output-name mashup.wav
```

# How to run batch evaluation

```bash
python scripts/batch_evaluate.py   --input-csv data/mashup_pairs.csv   --output-csv outputs/mashup_test.csv
```

## Credits / citations / license notes

This project uses open-source tools including yt-dlp, Demucs, librosa, torchaudio, pydub, scipy, and pedalboard. Used MIT license.
