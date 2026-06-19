from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pyloudnorm as pyln
from pedalboard import Compressor, Gain, HighShelfFilter, Limiter, LowShelfFilter, Pedalboard
from scipy import signal as scipy_signal
from scipy.signal import medfilt

from .download import ensure_dir
from .features import extract_segment_audio, read_wav
from .scoring import candidate_pairs_for_role
from .utils import CANONICAL_FLOW, ROLES, SEGMENT_POLICY, Segment, SongData, StemFeatures

MAX_STRETCH_PER_WINDOW = 0.04
TSM_BACKEND_PREF = "auto"


def safe_stft_params(num_samples: int, preferred_n_fft: int = 2048):
    if num_samples < 16:
        return None
    target = min(preferred_n_fft, num_samples)
    if target < 64:
        n_fft = max(16, target)
    else:
        n_fft = 2 ** int(np.floor(np.log2(target)))
    n_fft = min(n_fft, num_samples)
    if n_fft < 16:
        return None
    hop_length = max(1, n_fft // 4)
    win_length = n_fft
    return n_fft, hop_length, win_length, max(0, min(n_fft - hop_length, n_fft - 1))


def denoise_stem(y: np.ndarray, sr: int, stationary: bool = True, prop_decrease: float = 0.8) -> np.ndarray:
    if y is None:
        return y
    y = np.asarray(y)
    if y.size == 0:
        return y.astype(np.float32)
    if y.ndim == 2:
        chans = [denoise_stem(y[:, ch], sr, stationary=stationary, prop_decrease=prop_decrease) for ch in range(y.shape[1])]
        return np.stack(chans, axis=1).astype(np.float32)
    y = y.astype(np.float32, copy=False)
    params = safe_stft_params(len(y), preferred_n_fft=2048)
    if params is None:
        return y.astype(np.float32)
    n_fft, hop_length, win_length, _ = params
    try:
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        mag = np.abs(D)
        phase = np.angle(D)
        prop_decrease = float(np.clip(prop_decrease, 0.0, 1.0))
        floor_gain = 1.0 - prop_decrease
        if stationary:
            noise_len = min(max(int(0.25 * sr), 2 * n_fft), len(y))
            if noise_len <= hop_length * 2:
                return y.astype(np.float32)
            D_noise = librosa.stft(y[:noise_len], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            noise_profile = np.median(np.abs(D_noise), axis=1, keepdims=True)
            threshold = 1.5 * noise_profile
        else:
            noise_profile = np.median(mag, axis=1, keepdims=True)
            threshold = 1.25 * noise_profile
        gain = np.where(mag >= threshold, 1.0, floor_gain).astype(np.float32)
        mag_clean = mag * gain
        D_clean = mag_clean * np.exp(1j * phase)
        y_clean = librosa.istft(D_clean, hop_length=hop_length, win_length=win_length, length=len(y))
        return np.asarray(y_clean, dtype=np.float32)
    except Exception:
        return y.astype(np.float32)


def remove_timestretch_artifacts(y: np.ndarray, sr: int, blend_alpha: float = 0.3) -> np.ndarray:
    if y is None:
        return y
    y = np.asarray(y)
    if y.size == 0:
        return y.astype(np.float32)
    if y.ndim == 2:
        chans = [remove_timestretch_artifacts(y[:, ch], sr, blend_alpha=blend_alpha) for ch in range(y.shape[1])]
        return np.stack(chans, axis=1).astype(np.float32)
    y = y.astype(np.float32, copy=False)
    if len(y) < 3:
        return y.astype(np.float32)
    y_filtered = medfilt(y, kernel_size=3)
    blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    return (blend_alpha * y_filtered + (1.0 - blend_alpha) * y).astype(np.float32)


def spectral_smooth_boundary(seg1: np.ndarray, seg2: np.ndarray, sr: int, smooth_len_ms: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    seg1 = np.asarray(seg1, dtype=np.float32).flatten()
    seg2 = np.asarray(seg2, dtype=np.float32).flatten()
    if seg1.size == 0 or seg2.size == 0:
        return seg1, seg2
    smooth_samples = int(smooth_len_ms * sr / 1000)
    smooth_samples = min(smooth_samples, len(seg1) // 4, len(seg2) // 4)
    if smooth_samples < 16:
        return seg1, seg2
    seg1_end = seg1[-smooth_samples:]
    seg2_start = seg2[:smooth_samples]
    nperseg = min(256, len(seg1_end), len(seg2_start))
    if nperseg < 16:
        return seg1, seg2
    noverlap = min(nperseg // 2, nperseg - 1)
    try:
        _, _, Zxx1 = scipy_signal.stft(seg1_end, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, _, Zxx2 = scipy_signal.stft(seg2_start, fs=sr, nperseg=nperseg, noverlap=noverlap)
        if Zxx1.size == 0 or Zxx2.size == 0:
            return seg1, seg2
        t_frames = min(Zxx1.shape[1], Zxx2.shape[1])
        if t_frames < 1:
            return seg1, seg2
        Zxx1 = Zxx1[:, :t_frames]
        Zxx2 = Zxx2[:, :t_frames]
        alpha = np.linspace(1.0, 0.0, t_frames, dtype=np.float32)[np.newaxis, :]
        beta = np.linspace(0.0, 1.0, t_frames, dtype=np.float32)[np.newaxis, :]
        Zxx1_smooth = Zxx1 * alpha
        Zxx2_smooth = Zxx2 * beta
        _, seg1_end_smooth = scipy_signal.istft(Zxx1_smooth, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, seg2_start_smooth = scipy_signal.istft(Zxx2_smooth, fs=sr, nperseg=nperseg, noverlap=noverlap)
        seg1_end_smooth = librosa.util.fix_length(np.asarray(seg1_end_smooth, dtype=np.float32), size=smooth_samples)
        seg2_start_smooth = librosa.util.fix_length(np.asarray(seg2_start_smooth, dtype=np.float32), size=smooth_samples)
        return (
            np.concatenate([seg1[:-smooth_samples], seg1_end_smooth]).astype(np.float32),
            np.concatenate([seg2_start_smooth, seg2[smooth_samples:]]).astype(np.float32),
        )
    except Exception:
        return seg1, seg2


def rubberband_available() -> Optional[str]:
    import shutil
    return shutil.which("rubberband")


def time_stretch_audio_rubberband(y: np.ndarray, sr: int, rate: float) -> Optional[np.ndarray]:
    import subprocess
    import tempfile
    import soundfile as sf
    exe = rubberband_available()
    if exe is None:
        return None
    try:
        with tempfile.TemporaryDirectory() as td:
            in_wav = os.path.join(td, "in.wav")
            out_wav = os.path.join(td, "out.wav")
            sf.write(in_wav, y.astype(np.float32), sr)
            tempo_pct = rate * 100.0
            cmd = [exe, "-T", f"{tempo_pct:.6f}", "-c", "3", "-t", in_wav, out_wav]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            y_out, sr_out = sf.read(out_wav, dtype="float32", always_2d=False)
            if sr_out != sr:
                y_out = librosa.resample(y_out, orig_sr=sr_out, target_sr=sr)
            return np.asarray(y_out, dtype=np.float32)
    except Exception:
        return None


def time_stretch_audio(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    if len(y) == 0 or abs(rate - 1.0) < 1e-6:
        return y
    if TSM_BACKEND_PREF in ("auto", "rubberband"):
        y_rb = time_stretch_audio_rubberband(y, sr, rate)
        if y_rb is not None:
            return y_rb
    return librosa.effects.time_stretch(y, rate=rate)


def apply_local_tsm_with_beat_alignment(
    y: np.ndarray,
    sr: int,
    guideA: StemFeatures,
    guideB: StemFeatures,
    beats_a: np.ndarray,
    beats_b: np.ndarray,
    seg_start_time: float,
    apply_denoising: bool = True,
    apply_artifact_removal: bool = True,
    denoise_strength: float = 0.7,
    artifact_blend: float = 0.3,
) -> np.ndarray:
    if apply_denoising:
        y = denoise_stem(y, sr, stationary=True, prop_decrease=denoise_strength * 0.5)

    if guideA.tempo_med <= 0 or guideB.tempo_med <= 0 or not np.isfinite(guideA.tempo_med) or not np.isfinite(guideB.tempo_med):
        base_rate = 1.0
    else:
        base_rate = guideB.tempo_med / guideA.tempo_med
    base_rate = max(1.0 - MAX_STRETCH_PER_WINDOW, min(1.0 + MAX_STRETCH_PER_WINDOW, base_rate))

    if beats_a.size > 0 and beats_b.size > 0:
        seg_beats_a = beats_a[beats_a >= seg_start_time]
        seg_beats_b = beats_b[beats_b >= seg_start_time]
        if seg_beats_a.size > 1 and seg_beats_b.size > 1:
            beat_interval_a = np.median(np.diff(seg_beats_a[: min(4, len(seg_beats_a))]))
            beat_interval_b = np.median(np.diff(seg_beats_b[: min(4, len(seg_beats_b))]))
            if beat_interval_a > 0 and beat_interval_b > 0:
                alignment_rate = beat_interval_b / beat_interval_a
                final_rate = 0.7 * base_rate + 0.3 * alignment_rate
                final_rate = max(1.0 - MAX_STRETCH_PER_WINDOW, min(1.0 + MAX_STRETCH_PER_WINDOW, final_rate))
                base_rate = final_rate

    stretched = time_stretch_audio(y, sr, rate=base_rate)

    if apply_artifact_removal:
        stretched = remove_timestretch_artifacts(stretched, sr, blend_alpha=artifact_blend)
    if apply_denoising:
        stretched = denoise_stem(stretched, sr, stationary=True, prop_decrease=denoise_strength * 0.3)
    return stretched


def stereo_widen(audio: np.ndarray, sr: int, width: float = 1.3) -> np.ndarray:
    if len(audio) == 0:
        return audio
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    mid = (audio[:, 0] + audio[:, 1]) / 2
    side = (audio[:, 0] - audio[:, 1]) / 2
    side_widened = side * width
    left = mid + side_widened
    right = mid - side_widened
    stereo_out = np.stack([left, right], axis=1)
    peak = np.max(np.abs(stereo_out))
    if peak > 0.99:
        stereo_out = stereo_out * (0.99 / peak)
    return stereo_out.astype(np.float32)


def widen_stereo(audio: np.ndarray, width: float = 1.2) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio
    mid = 0.5 * (audio[:, 0] + audio[:, 1])
    side = 0.5 * (audio[:, 0] - audio[:, 1])
    side *= width
    out = np.stack([mid + side, mid - side], axis=1)
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def apply_mastering_chain(audio: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    if audio is None or audio.size == 0:
        return audio
    if audio.ndim == 1:
        audio_stereo = np.stack([audio, audio], axis=1)
    else:
        audio_stereo = audio
    try:
        board = Pedalboard([
            HighShelfFilter(cutoff_frequency_hz=3000.0, gain_db=1.5, q=0.7),
            LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=1.0, q=0.7),
            Compressor(threshold_db=-18.0, ratio=2.0, attack_ms=10.0, release_ms=80.0),
            Gain(gain_db=0.0),
            Limiter(threshold_db=-1.0, release_ms=50.0),
        ])
        effected = board(audio_stereo, sr)
        effected = widen_stereo(effected, width=1.2)
        meter = pyln.Meter(sr)
        current_loudness = meter.integrated_loudness(effected)
        if np.isfinite(current_loudness):
            normalized = pyln.normalize.loudness(effected, current_loudness, target_lufs)
        else:
            normalized = effected
        if audio.ndim == 1:
            return np.mean(normalized, axis=1).astype(np.float32)
        return normalized.astype(np.float32)
    except Exception:
        try:
            meter = pyln.Meter(sr)
            current_loudness = meter.integrated_loudness(audio)
            if np.isfinite(current_loudness):
                normalized = pyln.normalize.loudness(audio, current_loudness, target_lufs)
                return normalized.astype(np.float32)
        except Exception:
            pass
        return audio


def _pick_audio_and_feats(songA, songB, featsA, featsB, pick, role):
    if pick["song"] == "A":
        song_sel = songA
        feats_sel = featsA
    else:
        song_sel = songB
        feats_sel = featsB
    seg = next((s for s in song_sel.stretched_segments if s.idx == pick["seg_idx"]), None)
    return song_sel, feats_sel, seg


def render_segment_audio(choice: Dict, songA: SongData, songB: SongData, featsA, featsB, sr_out: int) -> np.ndarray:
    stems = choice.get("stems", {})
    if not stems:
        return np.zeros(0, dtype=np.float32)
    target_len = choice.get("target_length", 0.0)
    if target_len <= 0:
        target_len = max((seg.end - seg.start) for seg in songA.stretched_segments + songB.stretched_segments) if (songA.stretched_segments or songB.stretched_segments) else 0.0
    n_samples = max(1, int(round(target_len * sr_out)))
    mix = np.zeros(n_samples, dtype=np.float32)
    role_count = 0
    base_gain = {"drums": 0.8, "bass": 0.7, "other": 0.85, "vocals": 1.0}

    for role, pick in stems.items():
        song_sel, feats_sel, seg = _pick_audio_and_feats(songA, songB, featsA, featsB, pick, role)
        if seg is None:
            continue
        y = extract_segment_audio(song_sel.stretched[role], song_sel.stretched_sr, seg)
        f = feats_sel[(pick["seg_idx"], role)]
        beats_sel = song_sel.stretched_beats
        y_adj = apply_local_tsm_with_beat_alignment(
            y, song_sel.stretched_sr, f, f, beats_sel, beats_sel, seg.start,
            apply_denoising=True, apply_artifact_removal=True, denoise_strength=0.7, artifact_blend=0.3
        )
        L = min(len(mix), len(y_adj))
        if L == 0:
            continue
        rms_ref = float(np.sqrt(np.mean(np.square(y[:L])) + 1e-8))
        rms_y = float(np.sqrt(np.mean(np.square(y_adj[:L])) + 1e-8))
        gain = base_gain.get(role, 0.85)
        if rms_y > 0:
            gain *= (rms_ref / rms_y) ** 0.5
        mix[:L] += (gain * y_adj[:L]).astype(np.float32)
        role_count += 1

    if role_count == 0:
        return np.zeros(0, dtype=np.float32)
    peak = float(np.max(np.abs(mix))) if mix.size else 1.0
    if peak > 0.99:
        mix = 0.99 * mix / peak
    return mix.astype(np.float32)


def is_vocals_present_for_segment(feats: Dict[Tuple[int, str], StemFeatures], seg_idx: int, threshold: float = 0.005) -> bool:
    f = feats.get((seg_idx, "vocals"))
    if f is None:
        return False
    return bool(np.isfinite(f.rms_mean) and f.rms_mean > threshold)


def segment_exists_in_either(songA: SongData, songB: SongData, label: str) -> bool:
    label = label.lower()
    return any(s.label.lower() == label for s in songA.stretched_segments) or any(s.label.lower() == label for s in songB.stretched_segments)


def segments_by_label(song: SongData, label: str) -> List[Segment]:
    lbl = label.lower()
    return [s for s in song.stretched_segments if s.label.lower() == lbl]


def get_segment_by_idx(song: SongData, idx: int) -> Optional[Segment]:
    for s in song.stretched_segments:
        if s.idx == idx:
            return s
    return None


def build_segment_mix_for_label(
    label: str,
    songA: SongData,
    songB: SongData,
    pcsA,
    pcsB,
    featsA,
    featsB,
    lam: float,
    prefer_cross_song: bool = False,
    cached_label_mix: Optional[Dict[str, Dict]] = None,
) -> Optional[Dict]:
    label_l = label.lower()
    segsA = segments_by_label(songA, label_l)
    segsB = segments_by_label(songB, label_l)
    if not segsA and not segsB:
        return None

    policy = SEGMENT_POLICY.get(label_l, {"essential": set(), "optional": set(), "omitted": set()})
    essential = set(policy["essential"])
    optional = set(policy["optional"])
    omitted = set(policy["omitted"])

    if label_l == "verse":
        vocals_present_any = any(is_vocals_present_for_segment(featsA, s.idx) for s in segsA) or any(
            is_vocals_present_for_segment(featsB, s.idx) for s in segsB
        )
        if not vocals_present_any:
            essential = (essential - {"vocals"}) | {"other"}

    if cached_label_mix is not None and label_l in cached_label_mix:
        return cached_label_mix[label_l]

    role_best_pairs: Dict[str, Tuple[Tuple[int, int], float]] = {}
    for role in ROLES:
        if role in omitted:
            continue
        pairs = candidate_pairs_for_role(role, songA, songB, featsA, featsB, lam, top_k=4, allowed_labels={label_l})
        if pairs:
            role_best_pairs[role] = pairs[0]

    essentials_met = all(role in role_best_pairs for role in essential)
    chosen: Dict = {"label": label_l, "stems": {}, "target_length": 0.0}

    if not essentials_met:
        for role in essential:
            if role in role_best_pairs:
                ia, ib = role_best_pairs[role][0]
                fA = featsA.get((ia, role))
                fB = featsB.get((ib, role))
                if fA is not None and fB is not None:
                    song_sel = "A" if fA.rms_mean >= fB.rms_mean else "B"
                    chosen["stems"][role] = {"song": song_sel, "seg_idx": ia if song_sel == "A" else ib}
            else:
                candidates = []
                for s in segsA:
                    f = featsA.get((s.idx, role))
                    if f is not None:
                        candidates.append(("A", s.idx, f.rms_mean))
                for s in segsB:
                    f = featsB.get((s.idx, role))
                    if f is not None:
                        candidates.append(("B", s.idx, f.rms_mean))
                if candidates:
                    song_sel, seg_idx, _ = max(candidates, key=lambda c: c[2])
                    chosen["stems"][role] = {"song": song_sel, "seg_idx": seg_idx}

        for role in ROLES:
            if role in optional and role not in chosen["stems"]:
                cands = []
                for s in segsA:
                    f = featsA.get((s.idx, role))
                    if f is not None:
                        cands.append(("A", s.idx, f.rms_mean))
                for s in segsB:
                    f = featsB.get((s.idx, role))
                    if f is not None:
                        cands.append(("B", s.idx, f.rms_mean))
                if cands:
                    song_sel, seg_idx, _ = max(cands, key=lambda c: c[2])
                    chosen["stems"][role] = {"song": song_sel, "seg_idx": seg_idx}

        lengths = []
        for role, pick in chosen["stems"].items():
            seg_obj = get_segment_by_idx(songA if pick["song"] == "A" else songB, pick["seg_idx"])
            if seg_obj is not None:
                lengths.append(seg_obj.end - seg_obj.start)
        chosen["target_length"] = float(min(lengths) if lengths else 0.0)

        for role in omitted:
            chosen["stems"].pop(role, None)
        return chosen

    for role in ROLES:
        if role in omitted:
            continue
        if role in role_best_pairs:
            ia, ib = role_best_pairs[role][0]
            fA = featsA.get((ia, role))
            fB = featsB.get((ib, role))
            if fA is not None and fB is not None:
                song_sel = "A" if fA.rms_mean >= fB.rms_mean else "B"
                chosen["stems"][role] = {"song": song_sel, "seg_idx": ia if song_sel == "A" else ib}

    for role in omitted:
        chosen["stems"].pop(role, None)

    if label_l in ("chorus", "verse") and chosen["stems"]:
        chosen_stems = dict(chosen["stems"])
        if len({v["song"] for v in chosen_stems.values()}) == 1:
            alt_song = "B" if next(iter(chosen_stems.values()))["song"] == "A" else "A"
            for role in list(chosen_stems.keys()):
                candidates = []
                segs = segsB if alt_song == "B" else segsA
                feats = featsB if alt_song == "B" else featsA
                for s in segs:
                    f = feats.get((s.idx, role))
                    if f is not None:
                        candidates.append((alt_song, s.idx, f.rms_mean))
                if candidates:
                    song_sel, seg_idx, _ = max(candidates, key=lambda c: c[2])
                    chosen_stems[role] = {"song": song_sel, "seg_idx": seg_idx}
                    break
            chosen["stems"] = chosen_stems
    return chosen


def assemble_full_mashup(songA: SongData, songB: SongData, pcsA, pcsB, featsA, featsB, lam: float) -> np.ndarray:
    labels_present = [lbl for lbl in CANONICAL_FLOW if segment_exists_in_either(songA, songB, lbl)]
    cached_label_mix: Dict[str, Dict] = {}
    sr_out = songA.stretched_sr
    segments_audio = []

    for lbl in CANONICAL_FLOW:
        lbl_l = lbl.lower()
        if lbl_l not in labels_present:
            continue
        choice = cached_label_mix["chorus"] if (lbl_l == "chorus" and "chorus" in cached_label_mix) else build_segment_mix_for_label(
            lbl_l, songA, songB, pcsA, pcsB, featsA, featsB, lam, prefer_cross_song=True, cached_label_mix=None
        )
        if choice is None:
            continue
        if lbl_l == "chorus":
            cached_label_mix["chorus"] = choice
        audio_seg = render_segment_audio(choice, songA, songB, featsA, featsB, sr_out)
        if audio_seg.size > 0:
            segments_audio.append(audio_seg.astype(np.float32))

    if not segments_audio:
        return np.zeros(0, dtype=np.float32)

    rms_values = [np.sqrt(np.mean(seg**2) + 1e-8) for seg in segments_audio if seg.size]
    if rms_values:
        max_rms = max(rms_values)
        target_rms = 0.95 * max_rms
        balanced_segments = []
        for seg in segments_audio:
            rms = np.sqrt(np.mean(seg**2) + 1e-8)
            if rms < target_rms and rms > 0:
                seg = seg * (target_rms / rms)
            balanced_segments.append(seg.astype(np.float32))
        segments_audio = balanced_segments

    out = segments_audio[0]
    xf_len = int(0.02 * sr_out)
    fade_out = np.cos(0.5 * np.pi * np.linspace(0, 1, xf_len, dtype=np.float32))
    fade_in = np.sin(0.5 * np.pi * np.linspace(0, 1, xf_len, dtype=np.float32))

    for seg in segments_audio[1:]:
        out = np.asarray(out).flatten()
        seg = np.asarray(seg).flatten()
        out, seg = spectral_smooth_boundary(out, seg, sr_out, smooth_len_ms=30)
        if xf_len > 0 and len(out) > xf_len and len(seg) > xf_len:
            x1 = out[-xf_len:] * fade_out
            x2 = seg[:xf_len] * fade_in
            out = np.concatenate([out[:-xf_len], x1 + x2, seg[xf_len:]])
        else:
            out = np.concatenate([out, seg])

    return apply_mastering_chain(out, sr_out, target_lufs=-14.0)


def fallback_vocals_over_mashup(song1_dir: str, song2_dir: str, output_wav: str) -> None:
    sr = 44100
    stems1 = {stem: read_wav(Path(song1_dir) / f"{stem}.wav", sr) for stem in ["vocals", "other", "bass", "drums"]}
    stems2 = {stem: read_wav(Path(song2_dir) / f"{stem}.wav", sr) for stem in ["vocals", "other", "bass", "drums"]}
    base = stems1["other"]
    vocals = stems2["vocals"]
    L = min(len(base), len(vocals))
    mix = base[:L] * 0.85 + vocals[:L] * 0.9
    peak = float(np.max(np.abs(mix))) if mix.size else 1.0
    if peak > 0.99:
        mix = 0.99 * mix / peak
    ensure_dir(Path(output_wav).parent)
    import soundfile as sf
    sf.write(output_wav, mix.astype(np.float32), sr)
