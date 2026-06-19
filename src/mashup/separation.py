from __future__ import annotations

from pathlib import Path
from typing import Dict

import soundfile as sf
import torch
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model

from .download import ensure_dir


def _waveforms_to_stems(waveforms: torch.Tensor) -> torch.Tensor:
    if waveforms.dim() == 4:
        return waveforms[0]
    if waveforms.dim() == 3:
        return waveforms
    raise ValueError(f"Unexpected Demucs output shape: {tuple(waveforms.shape)}")


def extract_stems(song_path: str | Path, output_dir: str | Path) -> Dict[str, str]:
    output_dir = ensure_dir(output_dir)
    bag_of_models = pretrained.get_model("htdemucs")
    model = bag_of_models.models[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    waveform, sr = torchaudio.load(str(song_path))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    waveform = waveform.unsqueeze(0).to(device)

    with torch.no_grad():
        waveforms = apply_model(model, waveform)

    waveforms = _waveforms_to_stems(waveforms)
    stem_names = ["drums", "bass", "other", "vocals"]
    paths: Dict[str, str] = {}

    for i, stem in enumerate(stem_names):
        stem_audio = waveforms[i].detach().cpu().numpy().T
        file_path = str(Path(output_dir) / f"{stem}.wav")
        sf.write(file_path, stem_audio, sr)
        paths[stem] = file_path

    return paths
