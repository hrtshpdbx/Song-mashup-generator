from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


ROLES = ["drums", "bass", "other", "vocals"]
PITCHED_ROLES = {"bass", "vocals", "other"}

CANONICAL_FLOW = [
    "start", "intro", "verse", "chorus", "verse", "chorus",
    "bridge", "solo", "inst", "break", "chorus", "outro", "end",
]

SEGMENT_POLICY = {
    "start": {"essential": set(), "optional": {"other", "drums", "vocals", "bass"}, "omitted": set()},
    "intro": {"essential": {"other"}, "optional": {"drums", "bass", "vocals"}, "omitted": set()},
    "verse": {"essential": {"vocals"}, "optional": {"drums", "bass", "other"}, "omitted": set()},
    "chorus": {"essential": {"vocals", "other"}, "optional": {"drums", "bass"}, "omitted": set()},
    "bridge": {"essential": {"other"}, "optional": {"vocals", "drums", "bass"}, "omitted": set()},
    "solo": {"essential": {"drums", "bass", "other"}, "optional": {"vocals"}, "omitted": set()},
    "inst": {"essential": {"other"}, "optional": {"bass", "drums"}, "omitted": {"vocals"}},
    "break": {"essential": {"drums"}, "optional": {"other", "bass", "vocals"}, "omitted": set()},
    "outro": {"essential": {"other"}, "optional": {"drums", "bass", "vocals"}, "omitted": set()},
    "end": {"essential": set(), "optional": {"other", "drums", "vocals", "bass"}, "omitted": set()},
}

@dataclass
class Segment:
    start: float
    end: float
    label: str
    idx: int

@dataclass
class SongData:
    name: str
    dirpath: str
    bpm: float
    beats: np.ndarray
    downbeats: np.ndarray
    beat_positions: np.ndarray
    segments: List[Segment]
    sr: int
    audio_paths: Dict[str, str]
    stretched: Dict[str, np.ndarray] = field(default_factory=dict)
    stretched_sr: int = 0
    stretched_beats: np.ndarray = field(default_factory=lambda: np.array([]))
    stretched_downbeats: np.ndarray = field(default_factory=lambda: np.array([]))
    stretched_segments: List[Segment] = field(default_factory=list)

@dataclass
class StemFeatures:
    chroma_ti: np.ndarray
    mfcc: np.ndarray
    rms: np.ndarray
    rms_mean: float
    pitch_f0: Optional[np.ndarray]
    tempo_curve: np.ndarray
    tempo_med: float
    energy_curve: np.ndarray
    length_sec: float
    beats_count: int

@dataclass
class PrecomputedStem:
    S_mag: np.ndarray
    mel: np.ndarray
    mfcc: np.ndarray
    chroma: np.ndarray
    chroma_ti: np.ndarray
    onset_env: np.ndarray
    tempogram: np.ndarray
    tempo_curve: np.ndarray
    f0_hz: Optional[np.ndarray]
    rms: np.ndarray
    sr: int
    hop: int
    n_fft: int
    frame_times: np.ndarray
