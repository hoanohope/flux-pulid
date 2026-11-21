"""vision_core.py -- the project's standard image-generation function (handoff H1).

Quick start
-----------
    from src.vision_core import VisionEngine, DummyEngine, generate

    # On Kaggle (GPU): load the real model once (slow)
    engine = VisionEngine.from_yaml("configs/base.yaml", quant="Q4_0", layout="split")

    # On a machine without a GPU (for developing/testing pipelines):
    engine = DummyEngine.from_yaml("configs/base.yaml")

    res = generate({"prompt": "...", "ref_image": "ref.jpg", "seed": 42}, engine=engine)
    print(res.status, res.image_path, res.metadata["t_denoise_s"])

Error convention
----------------
- Invalid payload (wrong type/value)  -> raises ``pydantic.ValidationError`` (caller's mistake).
- Runtime failure (OOM, black image, PuLID not firing, ...) -> NOT raised; a
  ``GenerationResult`` with ``status != "OK"`` is returned so benchmark loops keep going.

status: OK | BLACK | OOM | ERROR
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.schema import (CONTRACT_VERSION, DEFAULT_SIZE, N_CA, GeneratePayload,
                        GenerationResult, RunOutput)
from src.utils import git_hash, load_config

__all__ = ["BaseEngine", "DummyEngine", "VisionEngine", "GeneratePayload", "GenerationResult",
           "RunOutput", "generate", "get_engine", "load_config"]

log = logging.getLogger("vision_core")


# ───────────────────────────── Base engine ─────────────────────────────
class BaseEngine:
    """Shared logic: payload validation, naming, saving, metadata logging, error classification."""

    engine_name = "base"
    quant = "n/a"
    layout = "n/a"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        std = cfg.get("standard") or {}
        self.default_w = std.get("width") or DEFAULT_SIZE
        self.default_h = std.get("height") or DEFAULT_SIZE
        paths = cfg.get("paths") or {}
        self.out_dir = Path(paths.get("out_dir", "outputs"))
        self.git = git_hash()

    @classmethod
    def from_yaml(cls, path: str = "configs/base.yaml", **kw):
        return cls(load_config(path), **kw)

    # --- implemented by subclasses ---
    def _run(self, p: GeneratePayload, w: int, h: int) -> RunOutput:
        raise NotImplementedError

    def _cleanup(self) -> None:
        gc.collect()

    # --- public API ---
    def generate(self, payload) -> GenerationResult:
        p = payload if isinstance(payload, GeneratePayload) else GeneratePayload(**payload)
        w, h = p.width or self.default_w, p.height or self.default_h
        expected = p.steps * N_CA if p.use_id else 0
        name = p.output_name or f"img_{int(time.time() * 1000)}_s{p.seed}"
        meta = {
            "contract_version": CONTRACT_VERSION, "engine": self.engine_name,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_hash": self.git, "quant": self.quant, "layout": self.layout,
            "name": name, "width": w, "height": h, "ca_expected": expected,
            "payload": p.model_dump(),
        }
        t_all = time.perf_counter()
        try:
            out = self._run(p, w, h)
        except Exception as e:                       # KeyboardInterrupt is deliberately not caught
            self._cleanup()
            status = "OOM" if type(e).__name__ == "OutOfMemoryError" else "ERROR"
            log.warning("generate %s: %s: %s", name, status, e)
            return self._finish(GenerationResult(status, None, None, f"{type(e).__name__}: {e}", meta),
                                t_all)

        meta.update({"t_id_s": round(out.t_id, 2), "t_encode_s": round(out.t_encode, 2),
                     "t_denoise_s": round(out.t_denoise, 2), "ca_fired": out.fired, **out.vram})
        status, err = "OK", None
        if self._is_black(out.image):
            status, err = "BLACK", "Black/NaN image (usually fp16 overflow)"
        elif out.fired != expected:
            status, err = "ERROR", f"PuLID fired {out.fired} times, expected {expected}"
        path = None
        if out.image is not None and status in ("OK", "BLACK"):
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = str(self.out_dir / f"{name}.png")
            out.image.save(path)
        return self._finish(GenerationResult(status, path, out.image, err, meta), t_all)

    def _finish(self, res: GenerationResult, t_all: float) -> GenerationResult:
        res.metadata["status"] = res.status
        res.metadata["error"] = res.error
        res.metadata["t_total_s"] = round(time.perf_counter() - t_all, 2)
        res.metadata["image_path"] = res.image_path
        try:                                         # one JSON line per call (failures included)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with open(self.out_dir / "metadata.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(res.metadata, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            log.warning("Could not write metadata.jsonl: %s", e)
        return res

    @staticmethod
    def _is_black(img) -> bool:
        if img is None:
            return False
        import numpy as np
        return bool(np.asarray(img).mean() < 5)


# ───────────────────────── Dummy engine (no GPU needed) ─────────────────────────
class DummyEngine(BaseEngine):
    """Returns a grey image with text; same interface and metadata as the real engine.
    Lets pipelines (benchmarks, schemas, tests) be developed on machines without a GPU."""

    engine_name = "dummy"
    quant = "dummy"
    layout = "dummy"

    def _run(self, p: GeneratePayload, w: int, h: int) -> RunOutput:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (w, h), (128, 128, 128))
        ImageDraw.Draw(img).text((10, 10), f"DUMMY seed={p.seed}\n{p.prompt[:80]}", fill=(255, 255, 255))
        return RunOutput(img, 0.0, 0.0, 0.0, p.steps * N_CA if p.use_id else 0,
                         {"vram_gpu0_gb": 0.0, "vram_gpu1_gb": 0.0})


# ───────────────────────────── Real engine ─────────────────────────────
FLUX_CONFIG = {
    "_class_name": "FluxTransformer2DModel", "_diffusers_version": "0.30.0",
    "attention_head_dim": 128, "guidance_embeds": True, "in_channels": 64,
    "joint_attention_dim": 4096, "num_attention_heads": 24, "num_layers": 19,
    "num_single_layers": 38, "patch_size": 1, "pooled_projection_dim": 768,
    "axes_dims_rope": [16, 56, 56],
}


class VisionEngine(BaseEngine):
    """FLUX.1-dev (GGUF) + PuLID on a dual-T4 machine (Kaggle). Load once, call ``generate()`` many times."""

    engine_name = "flux-gguf-pulid"
    LAYOUTS = ("1GPU", "split")

    def __init__(self, cfg: dict, quant: Optional[str] = None, layout: Optional[str] = None,
                 hf_token: Optional[str] = None):
        super().__init__(cfg)
        self.quant = quant or cfg["model"]["quant"]
        self.layout = layout or (cfg.get("standard") or {}).get("layout") or "split"
        if self.layout not in self.LAYOUTS:
            raise ValueError(f"layout must be one of {self.LAYOUTS}")
        if self.quant == "Q8_0" and self.layout == "1GPU":
            raise ValueError("Q8_0 supports only the 'split' layout (1GPU always runs out of memory on a T4)")
        self._hf_token = hf_token
        self._state = {"id": None, "weight": 1.0, "txt_len": 512, "fired": 0}
        self._hooks: list = []
        self._id_cache: dict = {}
        self._load()

    # ---------- model loading ----------
    def _token(self) -> Optional[str]:
        tok = self._hf_token or os.environ.get("HF_TOKEN")
        if tok:
            return tok
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            return None

    def _load_tokenizers(self, tok_dir: Path):
        from transformers import CLIPTokenizer, T5TokenizerFast
        if (tok_dir / "tokenizer").is_dir():
            return (CLIPTokenizer.from_pretrained(str(tok_dir / "tokenizer")),
                    T5TokenizerFast.from_pretrained(str(tok_dir / "tokenizer_2")))
        token = self._token()
        if not token:
            raise RuntimeError("No saved tokenizer and no HF_TOKEN (env var or Kaggle Secret) to download one")
        t1 = CLIPTokenizer.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="tokenizer", token=token)
        t2 = T5TokenizerFast.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="tokenizer_2", token=token)
        t1.save_pretrained(str(tok_dir / "tokenizer"))
        t2.save_pretrained(str(tok_dir / "tokenizer_2"))
        return t1, t2

    @staticmethod
    def _resolve_gguf(path: str) -> str:
        """Use ``path`` if it exists; otherwise search by file name under the nearest existing parent
        (datasets created from notebook output may keep sub-folders)."""
        p = Path(path)
        if p.exists():
            return str(p)
        for parent in p.parents:
            if parent.exists():
                hits = sorted(parent.rglob(p.name))
                if hits:
                    return str(hits[0])
                break
        raise FileNotFoundError(f"GGUF file not found: {path} (also searched by name '{p.name}')")

    def _load(self) -> None:
        import torch
        from diffusers import (FluxPipeline, FluxTransformer2DModel, AutoencoderKL,
                               FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig)
        from transformers import T5EncoderModel, CLIPTextModel, BitsAndBytesConfig
        self.torch = torch
        cfg, m = self.cfg, self.cfg["model"]
        dev, paths = cfg["devices"], cfg.get("paths") or {}
        self.flux_dev, self.txt_dev, self.id_dev = dev["flux"], dev["text"], dev["id"]
        # Q8_0 (~12.7 GB) plus T5 does not fit on one T4 -> load T5/CLIP straight onto the text GPU
        t5_load = dev.get("t5_load") or (self.txt_dev if self.quant == "Q8_0" else self.flux_dev)
        self.dtype = getattr(torch, m["dtype"])
        assert torch.cuda.device_count() >= 2, "Two GPUs required (e.g. Kaggle T4 x2)"

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        self.work_dir = paths.get("work_dir", "/kaggle/working")
        self.pulid_repo = paths.get("pulid_repo", "/kaggle/working/PuLID")
        tok_dir = Path(paths.get("tokenizers_dir", "/kaggle/working/flux_tokenizers"))
        gguf = self._resolve_gguf(m["gguf"][self.quant])

        cfg_dir = Path(self.work_dir) / "flux_transformer_config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text(json.dumps(FLUX_CONFIG, indent=2))
        tok1, tok2 = self._load_tokenizers(tok_dir)       # token errors surface before heavy loading
        gc.collect()
        torch.cuda.empty_cache()

        log.info("Loading FLUX %s ...", self.quant)
        transformer = FluxTransformer2DModel.from_single_file(
            gguf, config=str(cfg_dir), quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
            torch_dtype=self.dtype, local_files_only=True).to(self.flux_dev)
        t5_kwargs = {} if t5_load == self.flux_dev else {"device_map": {"": torch.device(t5_load).index}}
        t5 = T5EncoderModel.from_pretrained(
            f"{m['text_dir']}/text_encoder_2",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=self.dtype),
            torch_dtype=self.dtype, local_files_only=True, **t5_kwargs)
        clip = CLIPTextModel.from_pretrained(f"{m['text_dir']}/text_encoder", torch_dtype=self.dtype,
                                             local_files_only=True).to(t5_load)   # same device as T5
        vae = AutoencoderKL.from_pretrained(f"{m['text_dir']}/vae", torch_dtype=self.dtype,
                                            local_files_only=True).to(self.flux_dev)

        flux_dev = self.flux_dev

        class _FluxPipelineGPU0(FluxPipeline):             # always executes on flux_dev
            @property
            def _execution_device(self):
                return torch.device(flux_dev)

        self.pipe = _FluxPipelineGPU0(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae,
                                      text_encoder=clip, text_encoder_2=t5, tokenizer=tok1,
                                      tokenizer_2=tok2, transformer=transformer)
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        self._load_pulid()
        self._install_hooks()
        self._set_layout(self.layout)
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Engine ready: %s | layout %s", self.quant, self.layout)

    def _load_pulid(self) -> None:
        torch = self.torch
        from safetensors.torch import load_file
        sys.path.insert(0, self.pulid_repo)
        os.chdir(self.work_dir)                            # PuLID downloads antelopev2/EVA-CLIP into ./models
        from pulid.pipeline_flux import PuLIDPipeline
        ckpt = self.cfg["model"]["pulid"]["ckpt"]
        if not Path(ckpt).exists():
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download("guozinan/PuLID", "pulid_flux_v0.9.1.safetensors",
                                   local_dir=f"{self.work_dir}/models")
        self.pulid = PuLIDPipeline(self.pipe.transformer, self.id_dev, weight_dtype=torch.float16,
                                   onnx_provider="cpu")
        sd, groups = load_file(ckpt), {}
        for k, v in sd.items():
            mod, rest = k.split(".", 1)
            groups.setdefault(mod, {})[rest] = v
        for mod, s in groups.items():
            getattr(self.pulid, mod).load_state_dict(s, strict=True)
        del sd, groups
        gc.collect()
        self.pulid.pulid_ca.to(self.flux_dev).eval()
        self.pulid.pulid_encoder.eval()
        self.n_ca = len(self.pulid.pulid_ca)

    def _install_hooks(self) -> None:
        """Attach PuLID cross-attention after every 2nd double block and every 4th single block."""
        torch, st, pulid, tr = self.torch, self._state, self.pulid, self.pipe.transformer
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        def apply_ca(idx, img_tokens):
            st["fired"] += 1
            return img_tokens + st["weight"] * pulid.pulid_ca[idx](st["id"], img_tokens)

        def double_hook(idx):
            def hook(module, args, output):
                if st["id"] is None:
                    return output
                enc, hid = output
                return enc, apply_ca(idx, hid)
            return hook

        def single_hook(idx):
            def hook(module, args, output):
                if st["id"] is None:
                    return output
                if isinstance(output, tuple):
                    enc, hid = output
                    return enc, apply_ca(idx, hid)
                n = st["txt_len"]
                return torch.cat([output[:, :n], apply_ca(idx, output[:, n:])], dim=1)
            return hook

        def pre_hook(module, args, kwargs):
            e = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
            if e is not None:
                st["txt_len"] = e.shape[1]

        self._hooks.append(tr.register_forward_pre_hook(pre_hook, with_kwargs=True))
        for i, blk in enumerate(tr.transformer_blocks):
            if i % 2 == 0:
                self._hooks.append(blk.register_forward_hook(double_hook(i // 2)))
        n_double = len(self._hooks) - 1
        for j, blk in enumerate(tr.single_transformer_blocks):
            if j % 4 == 0:
                self._hooks.append(blk.register_forward_hook(single_hook(n_double + j // 4)))
        assert len(self._hooks) - 1 == self.n_ca, "Hook count differs from PuLID CA block count"

    # ---------- layout / measurement ----------
    def _text_device(self) -> str:
        return str(next(self.pipe.text_encoder_2.parameters()).device)

    def _set_layout(self, name: str) -> None:
        dev = {"1GPU": self.flux_dev, "split": self.txt_dev}[name]
        if self._text_device() != dev:
            self.pipe.text_encoder_2.to(dev)               # 4-bit T5 first (most failure-prone)
            self.pipe.text_encoder.to(dev)
            self._cleanup()
        self.layout = name

    def _sync(self) -> None:
        for d in range(2):
            self.torch.cuda.synchronize(d)

    def _cleanup(self) -> None:
        gc.collect()
        self.torch.cuda.empty_cache()

    def _peaks(self) -> dict:
        gb = 1024 ** 3
        return {f"vram_gpu{d}_gb": round(self.torch.cuda.max_memory_allocated(d) / gb, 2) for d in range(2)}

    # ---------- generation ----------
    def _id_embedding(self, ref_path: str):
        if ref_path in self._id_cache:
            return self._id_cache[ref_path]
        import numpy as np
        from PIL import Image
        img = Image.open(ref_path).convert("RGB")
        with self.torch.no_grad():
            emb, _ = self.pulid.get_id_embedding(np.array(img), cal_uncond=False)
        if self.torch.isnan(emb).any():
            raise ValueError("ID embedding is NaN (no face detected or fp16 overflow)")
        emb = emb.to(self.flux_dev, self.dtype)
        if len(self._id_cache) >= 16:
            self._id_cache.pop(next(iter(self._id_cache)))
        self._id_cache[ref_path] = emb
        return emb

    def _encode(self, text: str):
        torch, dev = self.torch, self._text_device()
        with torch.no_grad(), torch.cuda.device(torch.device(dev).index):
            pe, ppe, _ = self.pipe.encode_prompt(prompt=text, prompt_2=text, device=dev,
                                                 max_sequence_length=512)
        return pe.to(self.flux_dev, self.dtype), ppe.to(self.flux_dev, self.dtype)

    def _run(self, p: GeneratePayload, w: int, h: int) -> RunOutput:
        torch = self.torch
        if p.lora_path:
            raise NotImplementedError("LoRA is not supported yet (waiting for the G2 experiment results)")
        t_id, id_emb = 0.0, None
        if p.use_id:
            t0 = time.perf_counter()
            id_emb = self._id_embedding(p.ref_image)
            t_id = time.perf_counter() - t0
        for d in range(2):
            torch.cuda.reset_peak_memory_stats(d)
        self._sync()
        t0 = time.perf_counter()
        pe, ppe = self._encode(p.prompt)
        self._sync()
        t_enc = time.perf_counter() - t0
        self._state.update(id=id_emb, weight=p.id_weight, fired=0)
        t1 = time.perf_counter()
        try:
            img = self.pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe, height=h, width=w,
                            num_inference_steps=p.steps, guidance_scale=p.guidance,
                            generator=torch.Generator("cpu").manual_seed(p.seed)).images[0]
        finally:
            self._state["id"] = None                       # never leak an embedding into the next call
        self._sync()
        return RunOutput(img, t_id, t_enc, time.perf_counter() - t1, self._state["fired"], self._peaks())


# ───────────────────────────── Module-level helpers ─────────────────────────────
_ENGINE: Optional[BaseEngine] = None


def get_engine(config_path: str = "configs/base.yaml", dummy: bool = False, **kw) -> BaseEngine:
    """Create (or reuse) the process-wide engine."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = (DummyEngine if dummy else VisionEngine).from_yaml(config_path, **kw)
    return _ENGINE


def generate(payload, engine: Optional[BaseEngine] = None) -> GenerationResult:
    """Standard H1 entry point: ``generate(payload) -> GenerationResult``."""
    return (engine or get_engine()).generate(payload)
