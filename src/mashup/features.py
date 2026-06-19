from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np

from .utils import PITCHED_ROLES, ROLES, PrecomputedStem, Segment, SongData, StemFeatures


SR = 44100
N_FFT = 2048
HOP = 512
N_MFCC = 20
N_MELS = 64
USE_CQT_FOR_CHROMA = True
CACHE_DIR = ".mashup_cache"


def ensure_cache_dir(song_dir: str | Path) -> str:
    cdir = Path(song_dir) / CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    return str(cdir)


def read_wav(path: str | Path, target_sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return y


def load_original_audio_into_stretched(song: SongData) -> SongData:
    stretched = {}
    for role, path in song.audio_paths.items():
        stretched[role] = read_wav(path, song.sr)
    song.stretched = stretched
    song.stretched_sr = song.sr
    song.stretched_beats = song.beats
    song.stretched_downbeats = song.downbeats
    song.stretched_segments = song.segments
    return song


def stretch_array_times(times: np.ndarray, rate: float) -> np.ndarray:
    return times / rate if times.size else times


def stretch_segments(segments: list[Segment], rate: float) -> list[Segment]:
    return [Segment(start=s.start / rate, end=s.end / rate, label=s.label, idx=s.idx) for s in segments]


def beats_in_segment_stretched(song: SongData, seg: Segment) -> int:
    if song.stretched_beats.size == 0:
        return 0
    return int(np.sum((song.stretched_beats >= seg.start) & (song.stretched_beats < seg.end)))


def segment_to_frame_range(pc: PrecomputedStem, seg: Segment) -> Tuple[int, int]:
    t = pc.frame_times
    start_idx = int(np.searchsorted(t, seg.start, side="left"))
    end_idx = int(np.searchsorted(t, seg.end, side="left"))
    start_idx = max(0, min(start_idx, len(t)))
    end_idx = max(start_idx + 1, min(end_idx, len(t)))
    return start_idx, end_idx


def fix_length_vec(x: np.ndarray, n: int) -> np.ndarray:
    return librosa.util.fix_length(x, size=n)


def precompute_stem(song: SongData, role: str, fast_mode: bool = True) -> PrecomputedStem:
    cache_dir = ensure_cache_dir(song.dirpath)
    cache_key = f"{song.name}_{role}_sr{song.stretched_sr}_nfft{N_FFT}_hop{HOP}_nmfcc{N_MFCC}_mels{N_MELS}_cqt{int(USE_CQT_FOR_CHROMA)}.npz"
    cache_path = os.path.join(cache_dir, cache_key)

    if os.path.isfile(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        f0_arr = data["f0_hz"] if "f0_hz" in data.files and data["f0_hz"].size else None
        return PrecomputedStem(
            S_mag=data["S_mag"],
            mel=data["mel"],
            mfcc=data["mfcc"],
            chroma=data["chroma"],
            chroma_ti=data["chroma_ti"],
            onset_env=data["onset_env"],
            tempogram=data["tempogram"],
            tempo_curve=data["tempo_curve"],
            f0_hz=f0_arr,
            rms=data["rms"],
            sr=int(data["sr"]),
            hop=int(data["hop"]),
            n_fft=int(data["n_fft"]),
            frame_times=data["frame_times"],
        )

    y = song.stretched[role]
    sr = song.stretched_sr

    S_full = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    Hc, Pc = librosa.decompose.hpss(S_full)
    y_harm = librosa.istft(Hc, hop_length=HOP, length=len(y))
    y_perc = librosa.istft(Pc, hop_length=HOP, length=len(y))

    S = np.abs(S_full)
    S_harm = np.abs(librosa.stft(y_harm, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=N_MFCC)

    if USE_CQT_FOR_CHROMA:
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    else:
        chroma = librosa.feature.chroma_stft(S=S_harm, sr=sr, n_chroma=12)

    chroma_ti = np.max(np.stack([np.roll(chroma, k, axis=0) for k in range(12)], axis=0), axis=0)
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=HOP)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    tempi = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=HOP)
    tempo_curve = tempi[np.argmax(tempogram, axis=0)] if tempogram.size else np.zeros_like(onset_env)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP).flatten()

    f0_hz = None
    if role in PITCHED_ROLES:
        try:
            f0, _, _ = librosa.pyin(
                y_harm,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                frame_length=2048,
                hop_length=HOP,
            )
            f0_hz = np.nan_to_num(f0)
        except Exception:
            f0_hz = None

    n_frames = S.shape[1]
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=HOP)

    np.savez_compressed(
        cache_path,
        S_mag=S,
        mel=mel,
        mfcc=mfcc,
        chroma=chroma,
        chroma_ti=chroma_ti,
        onset_env=onset_env,
        tempogram=tempogram,
        tempo_curve=tempo_curve,
        f0_hz=(f0_hz if f0_hz is not None else np.array([])),
        rms=rms,
        sr=sr,
        hop=HOP,
        n_fft=N_FFT,
        frame_times=frame_times,
    )

    return PrecomputedStem(
        S_mag=S,
        mel=mel,
        mfcc=mfcc,
        chroma=chroma,
        chroma_ti=chroma_ti,
        onset_env=onset_env,
        tempogram=tempogram,
        tempo_curve=tempo_curve,
        f0_hz=f0_hz,
        rms=rms,
        sr=sr,
        hop=HOP,
        n_fft=N_FFT,
        frame_times=frame_times,
    )


def precompute_stem_worker(args):
    song, role, fast_mode = args
    return role, precompute_stem(song, role, fast_mode=fast_mode)


def precompute_all_stems(song: SongData, fast_mode: bool = True, parallel: bool = False) -> Dict[str, PrecomputedStem]:
    if parallel:
        tasks = [(song, r, fast_mode) for r in ROLES]
        with mp.Pool(processes=min(len(ROLES), max(1, mp.cpu_count() // 2))) as pool:
            results = pool.map(precompute_stem_worker, tasks)
    else:
        results = [precompute_stem_worker((song, r, fast_mode)) for r in ROLES]
    return {r: pc for r, pc in results}


def extract_segment_audio(y: np.ndarray, sr: int, seg: Segment) -> np.ndarray:
    a = int(round(seg.start * sr))
    b = int(round(seg.end * sr))
    a = max(0, min(a, len(y)))
    b = max(a, min(b, len(y)))
    return y[a:b]


def perceptual_loudness(y: np.ndarray, sr: int) -> float:
    import pyloudnorm as pyln

    if len(y) == 0:
        return -np.inf
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y.astype(np.float64))
        return float(loudness) if np.isfinite(loudness) else -100.0
    except Exception:
        rms = np.sqrt(np.mean(y**2) + 1e-8)
        return float(20 * np.log10(rms + 1e-8))


def build_segment_stem_features_stretched(song: SongData, seg: Segment, role: str, pc: PrecomputedStem) -> StemFeatures:
    a, b = segment_to_frame_range(pc, seg)
    if b - a < 2:
        a = max(0, a - 1)
        b = min(a + 2, pc.chroma_ti.shape[1])

    chroma_ti_seg = pc.chroma_ti[:, a:b] if pc.chroma_ti.size else np.zeros((12, 2), dtype=float)
    mfcc_seg = pc.mfcc[:, a:b] if pc.mfcc.size else np.zeros((N_MFCC, 2), dtype=float)
    rms_seg = pc.rms[a:b] if pc.rms.size else np.zeros(2, dtype=float)
    tempo_curve_seg = pc.tempo_curve[a:b] if pc.tempo_curve.size else np.zeros(2, dtype=float)

    f0_seg = None
    if pc.f0_hz is not None and isinstance(pc.f0_hz, np.ndarray) and pc.f0_hz.size:
        f0_seg = pc.f0_hz[a:b]

    seg_audio = extract_segment_audio(song.stretched[role], song.stretched_sr, seg)
    rms_mean = perceptual_loudness(seg_audio, song.stretched_sr)
    valid_tempo = tempo_curve_seg[tempo_curve_seg > 0]
    tempo_med = float(np.median(valid_tempo)) if valid_tempo.size else 0.0
    length_sec = max(1e-6, seg.end - seg.start)
    bcount = beats_in_segment_stretched(song, seg)

    return StemFeatures(
        chroma_ti=chroma_ti_seg,
        mfcc=mfcc_seg,
        rms=rms_seg,
        rms_mean=rms_mean,
        pitch_f0=f0_seg if (role in PITCHED_ROLES and f0_seg is not None and f0_seg.size) else None,
        tempo_curve=tempo_curve_seg,
        tempo_med=tempo_med,
        energy_curve=rms_seg,
        length_sec=length_sec,
        beats_count=bcount,
    )


def extract_all_stem_features_stretched(song: SongData, pcs: Dict[str, PrecomputedStem]) -> Dict[Tuple[int, str], StemFeatures]:
    feats = {}
    for seg in song.stretched_segments:
        for role in ROLES:
            feats[(seg.idx, role)] = build_segment_stem_features_stretched(song, seg, role, pcs[role])
    return feats
