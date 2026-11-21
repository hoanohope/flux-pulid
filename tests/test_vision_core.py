"""GPU-free tests for vision_core (payload, dummy engine, error classification, metadata).
Run:  python -m pytest tests -q
"""
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vision_core import (BaseEngine, DummyEngine, GeneratePayload, RunOutput, generate)


def make_engine(tmp_path, cls=DummyEngine, **std):
    cfg = {"paths": {"out_dir": str(tmp_path / "out")}, "standard": std}
    return cls(cfg)


# ---------- payload ----------
def test_payload_defaults_and_id_requires_ref():
    p = GeneratePayload(prompt="a", use_id=False)
    assert p.seed == 42 and p.steps == 20 and p.width is None
    with pytest.raises(ValidationError):
        GeneratePayload(prompt="a")                       # use_id defaults to True but ref_image is missing


@pytest.mark.parametrize("bad", [
    {"prompt": ""}, {"prompt": "a", "use_id": False, "width": 100},
    {"prompt": "a", "use_id": False, "width": 721}, {"prompt": "a", "use_id": False, "steps": 0},
    {"prompt": "a", "use_id": False, "guidance": 99}, {"prompt": "a", "use_id": False, "seed": -1},
    {"prompt": "a", "use_id": False, "output_name": "a/b"}, {"prompt": "a", "use_id": False, "typo_field": 1},
])
def test_payload_rejects_bad_values(bad):
    with pytest.raises(ValidationError):
        GeneratePayload(**bad)


# ---------- dummy engine ----------
def test_dummy_generate_ok_and_files(tmp_path):
    eng = make_engine(tmp_path, width=512, height=512)
    r = generate({"prompt": "hi", "ref_image": "x.jpg", "output_name": "t1", "seed": 7,
                  "tags": {"identity_id": "id01", "prompt_id": "p3"}}, engine=eng)
    assert r.status == "OK" and r.ok
    assert Path(r.image_path).exists() and r.image.size == (512, 512)     # size comes from the `standard` config
    m = r.metadata
    assert m["ca_fired"] == m["ca_expected"] == 20 * 20
    assert m["payload"]["tags"]["identity_id"] == "id01" and m["name"] == "t1"
    line = (tmp_path / "out" / "metadata.jsonl").read_text().strip().splitlines()
    assert len(line) == 1 and json.loads(line[0])["status"] == "OK"


def test_dummy_no_id_expects_zero_fired(tmp_path):
    r = generate({"prompt": "hi", "use_id": False, "width": 256, "height": 256}, engine=make_engine(tmp_path))
    assert r.status == "OK" and r.metadata["ca_expected"] == 0 and r.image.size == (256, 256)


def test_default_size_fallback_720(tmp_path):
    r = generate({"prompt": "hi", "use_id": False}, engine=make_engine(tmp_path))
    assert r.image.size == (720, 720)


def test_metadata_appends_one_line_per_call(tmp_path):
    eng = make_engine(tmp_path)
    for i in range(3):
        generate({"prompt": "hi", "use_id": False, "seed": i, "width": 256, "height": 256}, engine=eng)
    assert len((tmp_path / "out" / "metadata.jsonl").read_text().strip().splitlines()) == 3


# ---------- error classification (no GPU) ----------
class OutOfMemoryError(Exception):          # same class name as torch.cuda.OutOfMemoryError
    pass


class OOMEngine(BaseEngine):
    def _run(self, p, w, h):
        raise OutOfMemoryError("CUDA out of memory")


class BoomEngine(BaseEngine):
    def _run(self, p, w, h):
        raise RuntimeError("boom")


class BlackEngine(BaseEngine):
    def _run(self, p, w, h):
        from PIL import Image
        return RunOutput(Image.new("RGB", (64, 64), (0, 0, 0)), fired=0)


class WrongFiredEngine(BaseEngine):
    def _run(self, p, w, h):
        from PIL import Image
        return RunOutput(Image.new("RGB", (64, 64), (200, 200, 200)), fired=5)


def test_oom_returns_status_not_raise(tmp_path):
    r = generate({"prompt": "a", "use_id": False}, engine=make_engine(tmp_path, OOMEngine))
    assert r.status == "OOM" and r.image_path is None and "out of memory" in r.error
    assert (tmp_path / "out" / "metadata.jsonl").exists()          # failures are still logged


def test_runtime_error_becomes_error_status(tmp_path):
    r = generate({"prompt": "a", "use_id": False}, engine=make_engine(tmp_path, BoomEngine))
    assert r.status == "ERROR" and "boom" in r.error


def test_black_image_flagged(tmp_path):
    r = generate({"prompt": "a", "use_id": False}, engine=make_engine(tmp_path, BlackEngine))
    assert r.status == "BLACK"


def test_pulid_wrong_fired_count_is_error(tmp_path):
    r = generate({"prompt": "a", "ref_image": "x.jpg"}, engine=make_engine(tmp_path, WrongFiredEngine))
    assert r.status == "ERROR" and "PuLID" in r.error and r.image_path is None


def test_invalid_payload_raises(tmp_path):
    with pytest.raises(ValidationError):
        generate({"prompt": "a", "width": 99, "use_id": False}, engine=make_engine(tmp_path))
