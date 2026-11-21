"""Data contracts for the project (Pydantic models and result containers).

Two groups of schemas live here:

* Generation contract -- ``GeneratePayload`` (input of ``generate()``),
  ``GenerationResult`` and ``RunOutput`` (outputs).
* Experiment configuration -- ``PromptConfig`` (``configs/prompts.json``) and
  ``SelectionRecord`` (rows of ``configs/selection_log.csv``), as defined in
  ``docs/protocol.md``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_VERSION = "1.0"
DEFAULT_SIZE = 720   # used when neither the payload nor configs/base.yaml (``standard``) sets a size
N_CA = 20            # PuLID cross-attention blocks (10 double + 10 single)
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


# ───────────────────────── Generation contract ─────────────────────────
class GeneratePayload(BaseModel):
    """Validated input of :func:`src.vision_core.generate`."""

    model_config = ConfigDict(extra="forbid")      # a misspelled field fails immediately

    prompt: str = Field(min_length=1)
    ref_image: Optional[str] = None                # reference face path (required when use_id=True)
    use_id: bool = True                            # False = plain FLUX, no PuLID
    id_weight: float = Field(1.0, ge=0.0, le=2.0)
    seed: int = Field(42, ge=0, le=2**32 - 1)
    width: Optional[int] = None                    # None -> default from config
    height: Optional[int] = None
    steps: int = Field(20, ge=1, le=100)
    guidance: float = Field(3.5, ge=0.0, le=20.0)
    lora_path: Optional[str] = None                # placeholder: real engine does not support LoRA yet
    lora_scale: float = Field(1.0, ge=0.0, le=2.0)
    output_name: Optional[str] = None              # file stem; None -> auto-generated
    tags: dict[str, Any] = Field(default_factory=dict)   # identity_id, prompt_id, ... copied to metadata

    @field_validator("width", "height")
    @classmethod
    def _size_ok(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v % 16 != 0 or not (256 <= v <= 1536):
            raise ValueError("width/height must be divisible by 16 and within [256, 1536]")
        return v

    @field_validator("output_name")
    @classmethod
    def _name_ok(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _NAME_RE.match(v):
            raise ValueError("output_name may only contain letters, digits, '.', '_' and '-'")
        return v

    @model_validator(mode="after")
    def _id_needs_ref(self) -> "GeneratePayload":
        if self.use_id and not self.ref_image:
            raise ValueError("use_id=True requires ref_image")
        return self


@dataclass
class GenerationResult:
    """Outcome of one generation. Runtime failures are reported via ``status``, never raised."""

    status: str                                    # OK | BLACK | OOM | ERROR
    image_path: Optional[str] = None
    image: Any = None                              # PIL.Image (None on failure)
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> dict:
        return {"status": self.status, "image_path": self.image_path,
                "error": self.error, "metadata": self.metadata}


@dataclass
class RunOutput:
    """Raw output of an engine's ``_run`` (image plus timing and VRAM measurements)."""

    image: Any
    t_id: float = 0.0
    t_encode: float = 0.0
    t_denoise: float = 0.0
    fired: int = 0
    vram: dict = field(default_factory=dict)       # vram_gpu0_gb, vram_gpu1_gb


# ───────────────────── Experiment configuration (protocol.md) ─────────────────────
class PromptSpec(BaseModel):
    """One prompt entry of ``configs/prompts.json`` (protocol section 4.2)."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(pattern=r"^P\d{2,}$")
    style: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""


class PromptConfig(BaseModel):
    """Frozen prompt/seed configuration (protocol sections 4.3 - 4.6)."""

    model_config = ConfigDict(extra="forbid")

    version: str
    description: str = ""
    pilot_seeds: list[int] = Field(min_length=1)
    main_seeds: list[int] = Field(min_length=1)
    prompts: list[PromptSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "PromptConfig":
        ids = [p.prompt_id for p in self.prompts]
        if len(ids) != len(set(ids)):
            raise ValueError("prompt_id values must be unique")
        return self

    @classmethod
    def from_file(cls, path: str | Path = "configs/prompts.json") -> "PromptConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class SelectionRecord(BaseModel):
    """One row of ``configs/selection_log.csv`` (protocol section 3.4)."""

    model_config = ConfigDict(extra="forbid")

    image_path: str = Field(min_length=1)
    identity_id: str = Field(min_length=1)
    subset: Literal["train", "validation", "test"]
    selection_status: str
    reason: str = ""
