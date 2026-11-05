"""Tests for the All-In-One wrapper and local-audio inputs. No Docker, Cog, GPU or network needed:
subprocess calls are faked."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mashup import download, structure

REPO = Path(__file__).resolve().parents[1]


def _load_batch_module():
    spec = importlib.util.spec_from_file_location("batch_evaluate", REPO / "scripts" / "batch_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- structure.py
def test_select_structure_fields_keeps_beat_positions():
    data = {"bpm": 120, "beats": [0.5], "downbeats": [0.5], "beat_positions": [1], "segments": [], "extra": 1}
    out = structure.select_structure_fields(data)
    assert out == {"bpm": 120, "beats": [0.5], "downbeats": [0.5], "beat_positions": [1], "segments": []}


def test_select_structure_fields_accepts_camel_case():
    out = structure.select_structure_fields({"beatPositions": [1, 2, 3, 4]})
    assert out["beat_positions"] == [1, 2, 3, 4]


def test_allin1_image_default_and_override(monkeypatch):
    monkeypatch.delenv("MASHUP_ALLIN1_IMAGE", raising=False)
    assert structure.allin1_image() == "allin1-cu118"
    monkeypatch.setenv("MASHUP_ALLIN1_IMAGE", "my-image")
    assert structure.allin1_image() == "my-image"


def _fake_tools(monkeypatch, image_exists=True, calls=None):
    monkeypatch.setattr(structure.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, check=False, cwd=None, capture_output=False, text=False):
        if calls is not None:
            calls.append((list(cmd), cwd))
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0 if image_exists else 1, "", "")
        if cmd[:2] == ["cog", "predict"]:
            payload = {"bpm": 98.0, "beats": [0.1, 0.7], "downbeats": [0.1], "beat_positions": [1, 2],
                       "segments": [{"start": 0.0, "end": 10.0, "label": "chorus"}]}
            Path(cwd, "output.0.json").write_text(json.dumps(payload))
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(structure.subprocess, "run", fake_run)


def test_analyze_uses_local_image_absolute_path_and_clears_stale(monkeypatch, tmp_path):
    monkeypatch.delenv("MASHUP_ALLIN1_IMAGE", raising=False)
    monkeypatch.chdir(tmp_path)
    song_dir = Path("outputs/song1")  # relative on purpose: the capstone code broke on this
    song_dir.mkdir(parents=True)
    (song_dir / "original.wav").write_bytes(b"RIFF")
    (song_dir / "output.stale.json").write_text("{}")
    calls = []
    _fake_tools(monkeypatch, calls=calls)

    result = structure.analyze_audio_and_save_json(song_dir / "original.wav", song_dir, "song1.json")

    cog_cmd, cwd = next(c for c in calls if c[0][:2] == ["cog", "predict"])
    assert cog_cmd[2] == "allin1-cu118"
    input_path = cog_cmd[4].split("=@", 1)[1]
    assert Path(input_path).is_absolute() and Path(input_path).is_file()
    assert not (song_dir / "output.stale.json").exists()
    assert result["beat_positions"] == [1, 2]
    saved = json.loads((song_dir / "song1.json").read_text())
    assert saved["segments"][0]["label"] == "chorus" and saved["beat_positions"] == [1, 2]


def test_analyze_fails_clearly_when_image_missing(monkeypatch, tmp_path):
    (tmp_path / "original.wav").write_bytes(b"RIFF")
    _fake_tools(monkeypatch, image_exists=False)
    with pytest.raises(RuntimeError, match="build_allin1_image.sh"):
        structure.analyze_audio_and_save_json(tmp_path / "original.wav", tmp_path, "x.json")


def test_load_song_reads_old_and_new_json(tmp_path):
    for name in ["original", "vocals", "bass", "drums", "other"]:
        (tmp_path / f"{name}.wav").write_bytes(b"RIFF")
    seg = [{"start": 0.0, "end": 5.0, "label": "verse"}]
    (tmp_path / f"{tmp_path.name}.json").write_text(json.dumps(
        {"bpm": 100, "beats": [0.6], "downbeats": [0.6], "beatPositions": None, "segments": seg}))
    song = structure.load_song(tmp_path)
    assert song.beat_positions.size == 0 and song.segments[0].label == "verse"
    (tmp_path / f"{tmp_path.name}.json").write_text(json.dumps(
        {"bpm": 100, "beats": [0.6], "downbeats": [0.6], "beat_positions": [1], "segments": seg}))
    assert structure.load_song(tmp_path).beat_positions.tolist() == [1]


# ---------------------------------------------------------------- download.py
def test_is_local_audio(tmp_path):
    f = tmp_path / "abcdefghijk.webm"
    f.write_bytes(b"x")
    assert download.is_local_audio(f)
    assert not download.is_local_audio("https://youtu.be/abcdefghijk")
    assert not download.is_local_audio(tmp_path / "missing.webm")


def test_prepare_local_audio_calls_ffmpeg_at_44k(monkeypatch, tmp_path):
    src = tmp_path / "in.webm"
    src.write_bytes(b"x")
    seen = {}

    def fake_run(cmd, check=False):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"RIFF")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    out = download.get_audio(str(src), tmp_path / "song1")
    assert out.endswith("song1/original.wav") and Path(out).is_file()
    assert seen["cmd"][0] == "ffmpeg" and "44100" in seen["cmd"] and src.exists()


def test_youtube_download_has_no_forced_client():
    import inspect
    assert "player_client" not in inspect.getsource(download.download_youtube_audio)


# ---------------------------------------------------------------- batch_evaluate.py
def test_batch_resolves_column_aliases():
    batch = _load_batch_module()
    assert batch.resolve_columns(["song1", "song2", "mashup"]) == {"song1": "song1", "song2": "song2", "mashup": "mashup"}
    assert batch.resolve_columns(["row", "song1_id", "song2_id", "mashup_id", "usable"])["mashup"] == "mashup_id"
    assert batch.resolve_columns(["song1_url", "song2_url"])["mashup"] is None
    with pytest.raises(ValueError):
        batch.resolve_columns(["a", "b"])


def test_batch_resolves_ids_and_urls_to_downloaded_files(tmp_path):
    batch = _load_batch_module()
    (tmp_path / "dQw4w9WgXcQ.webm").write_bytes(b"x")
    expected = str(tmp_path / "dQw4w9WgXcQ.webm")
    assert batch.resolve_source("dQw4w9WgXcQ", tmp_path) == expected
    assert batch.resolve_source("https://youtu.be/dQw4w9WgXcQ?feature=shared", tmp_path) == expected
    assert batch.resolve_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=3", tmp_path) == expected
    assert batch.resolve_source(float("nan"), tmp_path) is None
    with pytest.raises(FileNotFoundError):
        batch.resolve_source("KvknOXGPzCQ", tmp_path)
    assert batch.resolve_source("KvknOXGPzCQ", None) == "https://www.youtube.com/watch?v=KvknOXGPzCQ"


def test_batch_skips_unusable_records_errors_and_resumes(monkeypatch, tmp_path):
    batch = _load_batch_module()
    audio = tmp_path / "audio"
    audio.mkdir()
    for vid in ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc", "ddddddddddd"]:
        (audio / f"{vid}.webm").write_bytes(b"x")
    csv = tmp_path / "triples.csv"
    csv.write_text("row,song1_id,song2_id,mashup_id,usable\n"
                   "0,aaaaaaaaaaa,bbbbbbbbbbb,ccccccccccc,1\n"
                   "1,aaaaaaaaaaa,bbbbbbbbbbb,ccccccccccc,0\n"
                   "2,aaaaaaaaaaa,eeeeeeeeeee,ccccccccccc,1\n"
                   "3,ddddddddddd,bbbbbbbbbbb,,1\n")
    calls = []

    def fake_main(s1, s2, output_dir, output_name, human_mashup_url=None, return_results=False):
        calls.append((s1, s2, human_mashup_url))
        ev = {"final_score": 0.4, "chroma_harmonic_similarity": 0.1, "clap_embedding_similarity": 0.5,
              "technical_peak_and_snr_score": 1.0, "final_snr": -2.0, "peak": 0.9}
        return {"machine_eval": ev, "human_eval": ev if human_mashup_url else None,
                "processing_time": 1.0, "output_path": str(Path(output_dir) / output_name)}

    import types
    fake_cli = types.ModuleType("mashup.cli")
    fake_cli.main = fake_main
    monkeypatch.setitem(sys.modules, "mashup.cli", fake_cli)

    out = tmp_path / "res.csv"
    batch.run_batch(csv, out, tmp_path / "runs", audio_dir=audio)
    import pandas as pd
    res = pd.read_csv(out).set_index("row_index")
    assert list(res.index) == [0, 2, 3]                       # row 1 skipped (usable=0)
    assert res.loc[0, "status"] == "ok" and res.loc[0, "human_final_score"] == 0.4
    assert res.loc[2, "status"] == "error" and "eeeeeeeeeee" in res.loc[2, "error"]
    assert res.loc[3, "status"] == "ok" and pd.isna(res.loc[3, "human_final_score"])
    assert len(calls) == 2

    batch.run_batch(csv, out, tmp_path / "runs", audio_dir=audio)  # resume: only the failed row is retried
    assert len(calls) == 2
    assert list(pd.read_csv(out)["row_index"]) == [0, 2, 3]
