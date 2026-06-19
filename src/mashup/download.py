from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path

import yt_dlp
from pydub import AudioSegment


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
    if ext in {".mp3", ".m4a", ".aac", ".flac", ".ogg"}:
        wav_path = str(Path(audio_path).with_suffix(".wav"))
        subprocess.run(["ffmpeg", "-y", "-i", audio_path, wav_path], check=True)
        return wav_path
    raise ValueError(f"Unsupported audio type: {audio_path}")


def download_youtube_audio(url: str, target_folder: str | Path, base_filename: str) -> str:
    target_folder = ensure_dir(target_folder)
    clean_url = normalize_youtube_url(url)
    output_template = str(target_folder / f"{base_filename}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
        "extractor_args": {"youtube": {"player_client": ["android"]}},
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
