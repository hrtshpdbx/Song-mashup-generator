from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path

import yt_dlp
from pydub import AudioSegment


AUDIO_EXTENSIONS = {".mp3", ".m4a", ".aac", ".flac", ".ogg", ".webm", ".opus"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_youtube_url(url: str) -> str:
    url = url.strip()
    match = re.match(r"https?://youtu\.be/([^?\s&]+)", url)
    if match:
        return f"https://www.youtube.com/watch?v={match.group(1)}"
    match = re.search(r"v=([^\s&#]+)", url)
    if match:
        return f"https://www.youtube.com/watch?v={match.group(1)}"
    return url


def convert_to_wav_if_needed(audio_path: str | Path) -> str:
    audio_path = str(audio_path)
    ext = Path(audio_path).suffix.lower()
    if ext == ".wav":
        return audio_path
    if ext in AUDIO_EXTENSIONS:
        wav_path = str(Path(audio_path).with_suffix(".wav"))
        subprocess.run(["ffmpeg", "-y", "-i", audio_path, wav_path], check=True)
        return wav_path
    raise ValueError(f"Unsupported audio type: {audio_path}")


def is_local_audio(source: str | Path) -> bool:
    """True if `source` is an existing audio file rather than a URL."""
    p = Path(str(source)).expanduser()
    return p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS | {".wav"}


def prepare_local_audio(source: str | Path, target_folder: str | Path, base_filename: str = "original") -> str:
    """Decode a local audio file (e.g. a .webm/Opus file saved by tools/fetch_audio.py) to
    <target_folder>/<base_filename>.wav at 44.1 kHz stereo, the rate HT-Demucs and the rest of
    the pipeline expect. The source file is left untouched."""
    target_folder = ensure_dir(target_folder)
    src = Path(str(source)).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Audio file not found: {src}")
    wav_path = (target_folder / f"{base_filename}.wav").resolve()
    if src == wav_path:
        return str(wav_path)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ar", "44100", "-ac", "2", str(wav_path)],
        check=True,
    )
    return str(wav_path)


def get_audio(source: str, target_folder: str | Path, base_filename: str = "original") -> str:
    """Local audio file or YouTube URL -> <target_folder>/original.wav."""
    if is_local_audio(source):
        return prepare_local_audio(source, target_folder, base_filename)
    return download_youtube_audio(source, target_folder, base_filename)


def download_youtube_audio(url: str, target_folder: str | Path, base_filename: str) -> str:
    target_folder = ensure_dir(target_folder)
    clean_url = normalize_youtube_url(url)
    output_template = str(target_folder / f"{base_filename}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
        # No forced player client or User-Agent: yt-dlp's defaults track YouTube's changes; the old
        # "android" client override now leads to HTTP 403 errors.
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([clean_url])

    downloaded_file = next(
        (f for f in glob.glob(str(target_folder / f"{base_filename}.*")) if f.endswith(".m4a")),
        None,
    )
    if not downloaded_file:
        raise FileNotFoundError(f"No audio file found for {url}")

    wav_path = str(target_folder / "original.wav")
    audio = AudioSegment.from_file(downloaded_file, format="m4a")
    audio.export(wav_path, format="wav")
    os.remove(downloaded_file)
    return wav_path
