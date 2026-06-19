from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import numpy as np

from .features import extract_segment_audio, read_wav
from .structure import load_index_json
from .utils import Segment

try:
    import laion_clap
    CLAP_AVAILABLE = True
except Exception:
    CLAP_AVAILABLE = False


def audio_peak(y):
    return float(np.max(np.abs(y))) if len(y) else 0.0


def audio_snr(reference, estimate):
    n = min(len(reference), len(estimate))
    reference = reference[:n]
    estimate = estimate[:n]
    if len(reference) == 0 or len(estimate) == 0:
        return 0.0
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((reference - estimate) ** 2)
    if noise_power <= 1e-9:
        return 100.0
    return float(10 * np.log10(signal_power / noise_power + 1e-12))


def get_clap_embedding(audio_path):
    if not CLAP_AVAILABLE:
        return None
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    embed = model.get_audio_embedding_from_filelist([audio_path], use_tensor=False)
    return embed[0]


def cos_sim(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def _segments_from_meta(meta: dict):
    return [Segment(start=float(seg["start"]), end=float(seg["end"]), label=seg.get("label", "segment"), idx=i) for i, seg in enumerate(meta.get("segments", []))]


def segmentwise_chroma(y_m, y_ref, sr, meta):
    segs = _segments_from_meta(meta)
    scores = []
    for seg in segs:
        m = extract_segment_audio(y_m, sr, seg)
        r = extract_segment_audio(y_ref, sr, seg)
        if len(m) < sr // 10 or len(r) < sr // 10:
            continue
        c_m = librosa.feature.chroma_cqt(y=m, sr=sr)
        c_r = librosa.feature.chroma_cqt(y=r, sr=sr)
        scores.append(float(np.mean(np.sum(c_m * c_r, axis=0) / (np.linalg.norm(c_m, axis=0) * np.linalg.norm(c_r, axis=0) + 1e-8))))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_mashup(json1_path, json2_path, audio1_path, audio2_path, mashup_path):
    meta1 = load_index_json(json1_path)
    meta2 = load_index_json(json2_path)
    y1 = read_wav(audio1_path, 44100)
    y2 = read_wav(audio2_path, 44100)
    y_m = read_wav(mashup_path, 44100)
    sr = 44100

    chroma_sim1 = segmentwise_chroma(y_m, y1, sr, meta1)
    chroma_sim2 = segmentwise_chroma(y_m, y2, sr, meta2)
    chromagram_eval = float(np.mean([chroma_sim1, chroma_sim2]))

    if CLAP_AVAILABLE:
        emb_m = get_clap_embedding(mashup_path)
        emb1 = get_clap_embedding(audio1_path)
        emb2 = get_clap_embedding(audio2_path)
        clap_sim = float(np.mean([cos_sim(emb_m, emb1), cos_sim(emb_m, emb2)]))
    else:
        clap_sim = None

    peak_m = audio_peak(y_m)
    snr1 = audio_snr(y1, y_m)
    snr2 = audio_snr(y2, y_m)
    tech_score = float(1.0 - max(peak_m - 0.99, 0))

    scores = [chromagram_eval, np.mean([snr1, snr2]) / 100.0, tech_score]
    weights = [0.35, 0.15, 0.15]
    if clap_sim is not None:
        scores.insert(1, clap_sim)
        weights.insert(1, 0.35)
    final_score = float(np.sum([w * s for w, s in zip(weights, scores)]) / np.sum(weights))

    return {
        "chroma_harmonic_similarity": float(chromagram_eval),
        "clap_embedding_similarity": None if clap_sim is None else float(clap_sim),
        "technical_peak_and_snr_score": float(tech_score),
        "final_snr": float(np.mean([snr1, snr2])),
        "final_score": float(final_score),
        "peak": float(peak_m),
        "snr_song1": float(snr1),
        "snr_song2": float(snr2),
    }
