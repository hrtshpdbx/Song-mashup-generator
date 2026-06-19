from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import fix_length_vec
from .utils import ROLES, Segment, SongData, StemFeatures


W_H = 0.40
W_T = 0.20
W_E = 0.10
W_TAU = 0.35
W_C = 0.25
W_P = 0.25
W_T_DRUM = 0.35
W_E_DRUM = 0.15
W_TAU_DRUM = 0.35
W_C_DRUM = 0.25
ALPHA_STRUCT = 0.25
BETA_LEN = 0.35
GAMMA_ECURVE = 0.25
DELTA_TAU_CONT = 0.20
TOP_K_PER_ROLE = 4
BEAM_WIDTH = 5


def normalize_curve(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    m = np.mean(x)
    s = np.std(x)
    if not np.isfinite(m) or not np.isfinite(s) or s < 1e-9:
        return np.zeros_like(x, dtype=float)
    y = (x - m) / s
    y[~np.isfinite(y)] = 0.0
    return y


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    val = float(np.dot(a, b) / (na * nb))
    if not np.isfinite(val):
        return 0.0
    return max(-1.0, min(1.0, val))


def framewise_cosine(A: np.ndarray, B: np.ndarray) -> float:
    if A.ndim == 2 and A.size:
        muA = np.mean(A, axis=1)
    else:
        muA = A if A.ndim == 1 else np.zeros(1)
    if B.ndim == 2 and B.size:
        muB = np.mean(B, axis=1)
    else:
        muB = B if B.ndim == 1 else np.zeros(1)
    muA = np.nan_to_num(muA, nan=0.0, posinf=0.0, neginf=0.0)
    muB = np.nan_to_num(muB, nan=0.0, posinf=0.0, neginf=0.0)
    return cosine_similarity(muA, muB)


def energy_similarity(e1: float, e2: float) -> float:
    if not np.isfinite(e1) or not np.isfinite(e2):
        return 0.0
    if e1 <= 1e-9 and e2 <= 1e-9:
        return 1.0
    return float(min(e1, e2) / max(e1, e2))


def tempo_avg_compat(t1: float, t2: float) -> float:
    if not np.isfinite(t1) or not np.isfinite(t2) or t1 <= 0 or t2 <= 0:
        return 0.5
    return float(min(t1, t2) / max(t1, t2))


def tempo_continuity(c1: np.ndarray, c2: np.ndarray) -> float:
    n = 128
    x1 = normalize_curve(fix_length_vec(np.nan_to_num(c1, nan=0.0, posinf=0.0, neginf=0.0), n))
    x2 = normalize_curve(fix_length_vec(np.nan_to_num(c2, nan=0.0, posinf=0.0, neginf=0.0), n))
    if x1.size == 0 or x2.size == 0:
        return 0.0
    return cosine_similarity(x1, x2)


def pitch_contour_similarity(f01: Optional[np.ndarray], f02: Optional[np.ndarray]) -> float:
    if f01 is None or f02 is None:
        return 0.0
    x1 = np.asarray(f01, dtype=float)
    x2 = np.asarray(f02, dtype=float)
    if x1.size == 0 or x2.size == 0:
        return 0.0
    n = 128
    x1 = fix_length_vec(np.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0), n)
    x2 = fix_length_vec(np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0), n)

    def hz_to_cents(x):
        x = np.maximum(x, 1e-6)
        valid = x[x > 0]
        med = float(np.median(valid)) if valid.size else 1.0
        c = 1200.0 * np.log2(x / med)
        c[~np.isfinite(c)] = 0.0
        return c

    c1 = normalize_curve(hz_to_cents(x1))
    c2 = normalize_curve(hz_to_cents(x2))
    return cosine_similarity(c1, c2)


def harmonic_similarity_chroma_ti(C1: np.ndarray, C2: np.ndarray) -> float:
    return framewise_cosine(C1, C2)


def length_match_score(l1: float, l2: float) -> float:
    if l1 <= 0 or l2 <= 0:
        return 0.0
    return float(min(l1, l2) / max(l1, l2))


def structure_match_score(lbl1: str, lbl2: str) -> float:
    a, b = lbl1.lower(), lbl2.lower()
    if a == b:
        return 1.0
    similar = [
        {"chorus", "hook", "refrain", "drop"},
        {"verse", "rap", "vocal"},
        {"intro", "build"},
        {"outro", "ending", "fade"},
        {"bridge", "pre-chorus", "build"},
    ]
    for g in similar:
        if a in g and b in g:
            return 0.7
    return 0.3


def vertical_stem_compat(role: str, fA: StemFeatures, fB: StemFeatures) -> float:
    if role == "drums":
        T = framewise_cosine(fA.mfcc, fB.mfcc)
        E = energy_similarity(fA.rms_mean, fB.rms_mean)
        Rtau = tempo_avg_compat(fA.tempo_med, fB.tempo_med)
        Ctau = tempo_continuity(fA.tempo_curve, fB.tempo_curve)
        return float(W_T_DRUM * T + W_E_DRUM * E + W_TAU_DRUM * Rtau + W_C_DRUM * Ctau)
    H = harmonic_similarity_chroma_ti(fA.chroma_ti, fB.chroma_ti)
    T = framewise_cosine(fA.mfcc, fB.mfcc)
    E = energy_similarity(fA.rms_mean, fB.rms_mean)
    Rtau = tempo_avg_compat(fA.tempo_med, fB.tempo_med)
    Ctau = tempo_continuity(fA.tempo_curve, fB.tempo_curve)
    P = pitch_contour_similarity(fA.pitch_f0, fB.pitch_f0)
    total_w = W_H + W_T + W_E + W_TAU + W_C + W_P
    if total_w <= 1e-9 or not np.isfinite(total_w):
        total_w = 1.0
    return float((W_H * H + W_T * T + W_E * E + W_TAU * Rtau + W_C * Ctau + W_P * P) / total_w)


def horizontal_stem_compat(segA: Segment, segB: Segment, fA: StemFeatures, fB: StemFeatures) -> float:
    S = structure_match_score(segA.label, segB.label)
    L = length_match_score(fA.length_sec, fB.length_sec)
    Ecurve = tempo_continuity(fA.energy_curve, fB.energy_curve)
    Ctau = tempo_continuity(fA.tempo_curve, fB.tempo_curve)
    return float(ALPHA_STRUCT * S + BETA_LEN * L + GAMMA_ECURVE * Ecurve + DELTA_TAU_CONT * Ctau)


def combined_stem_score(role: str, segA: Segment, segB: Segment, fA: StemFeatures, fB: StemFeatures, lam: float = 0.6) -> float:
    MV = vertical_stem_compat(role, fA, fB)
    MH = horizontal_stem_compat(segA, segB, fA, fB)
    s = float(lam * MV + (1.0 - lam) * MH)
    return 0.0 if not np.isfinite(s) else s


def candidate_pairs_for_role(
    role: str,
    songA: SongData,
    songB: SongData,
    featsA: Dict[Tuple[int, str], StemFeatures],
    featsB: Dict[Tuple[int, str], StemFeatures],
    lam: float,
    top_k: int,
    allowed_labels: Optional[set] = None,
) -> List[Tuple[Tuple[int, int], float]]:
    scores = []
    for sa in songA.stretched_segments:
        if allowed_labels and sa.label.lower() not in allowed_labels:
            continue
        fA = featsA[(sa.idx, role)]
        for sb in songB.stretched_segments:
            if allowed_labels and sb.label.lower() not in allowed_labels:
                continue
            fB = featsB[(sb.idx, role)]
            s = combined_stem_score(role, sa, sb, fA, fB, lam=lam)
            scores.append(((sa.idx, sb.idx), float(0.0 if not np.isfinite(s) else s)))
    if not scores:
        return []
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k] if top_k > 0 else scores


def candidate_pairs_for_role_global(
    role: str,
    songA: SongData,
    songB: SongData,
    featsA,
    featsB,
    lam: float,
    top_k: int,
) -> List[Tuple[Tuple[int, int], float]]:
    scores = []
    for sa in songA.stretched_segments:
        fA = featsA[(sa.idx, role)]
        for sb in songB.stretched_segments:
            fB = featsB[(sb.idx, role)]
            s = combined_stem_score(role, sa, sb, fA, fB, lam=lam)
            scores.append(((sa.idx, sb.idx), float(0.0 if not np.isfinite(s) else s)))
    if not scores:
        return []
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k] if top_k > 0 else scores


def quartet_penalty_tsm_and_consistency(assign, featsA, featsB) -> float:
    tempo_vals = []
    for role, (ia, ib) in assign.items():
        fA = featsA.get((ia, role))
        fB = featsB.get((ib, role))
        if fA is None or fB is None:
            continue
        tempo_vals.extend([fA.tempo_med, fB.tempo_med])
    tempo_vals = [x for x in tempo_vals if np.isfinite(x) and x > 0]
    if not tempo_vals:
        return 0.0
    med = float(np.median(tempo_vals))
    dev = np.mean([abs(x - med) / max(1.0, med) for x in tempo_vals])
    return float(dev) if np.isfinite(dev) else 0.0


def select_quartet(songA: SongData, songB: SongData, featsA, featsB, lam: float = 0.6, beam_width: int = BEAM_WIDTH, top_k_per_role: int = TOP_K_PER_ROLE):
    top_by_role = {}
    for role in ROLES:
        lst = candidate_pairs_for_role_global(role, songA, songB, featsA, featsB, lam, top_k_per_role)
        if not lst and songA.stretched_segments and songB.stretched_segments:
            lst = [((songA.stretched_segments[0].idx, songB.stretched_segments[0].idx), 0.0)]
        top_by_role[role] = lst

    beams = [({}, 0.0)]
    for role in ROLES:
        candidates = top_by_role[role]
        if not candidates:
            return None, -1e9
        new_beams = []
        for assign, score in beams:
            for (ia_ib, s) in candidates:
                new_assign = dict(assign)
                new_assign[role] = ia_ib
                new_beams.append((new_assign, score + s))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width] if new_beams else beams

    best, best_score = None, -1e9
    for assign, score in beams:
        penalty = quartet_penalty_tsm_and_consistency(assign, featsA, featsB)
        adj = score - 2.0 * penalty
        if adj > best_score and np.isfinite(adj):
            best_score, best = adj, assign
    return best, best_score
