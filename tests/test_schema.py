"""Tests for configuration schemas and utilities (CPU only)."""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.schema import PromptConfig, SelectionRecord
from src.utils import load_config, read_selection_log, retry


def test_prompts_json_is_valid_and_frozen_ids_unique():
    cfg = PromptConfig.from_file(ROOT / "configs" / "prompts.json")
    assert cfg.version and len(cfg.prompts) >= 1
    assert len({p.prompt_id for p in cfg.prompts}) == len(cfg.prompts)


def test_prompt_config_rejects_duplicate_ids():
    p = {"prompt_id": "P01", "style": "anime", "prompt": "x"}
    with pytest.raises(ValidationError):
        PromptConfig(version="1", pilot_seeds=[1], main_seeds=[1], prompts=[p, p])


def test_base_yaml_loads_and_has_required_sections():
    cfg = load_config(ROOT / "configs" / "base.yaml")
    for key in ("model", "devices", "gen", "standard"):
        assert key in cfg
    assert cfg["standard"]["width"] % 16 == 0


def test_selection_log_rows_are_valid_and_identity_disjoint():
    seen: dict[str, set[str]] = {}
    for rec in read_selection_log(ROOT / "configs" / "selection_log.csv"):
        seen.setdefault(rec.subset, set()).add(rec.identity_id)
    # identity-level split: no identity may appear in two subsets (no train/test leakage)
    assert not (seen["train"] & seen["test"])
    assert not (seen["train"] & seen["validation"])
    assert not (seen["validation"] & seen["test"])


def test_selection_record_rejects_unknown_subset():
    with pytest.raises(ValidationError):
        SelectionRecord(image_path="a.jpg", identity_id="1", subset="holdout", selection_status="included")


def test_retry_retries_then_succeeds():
    calls = {"n": 0}

    @retry(times=3, delay=0, exceptions=(ValueError,))
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    assert flaky() == "ok" and calls["n"] == 3
