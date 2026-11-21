"""Minimal end-to-end demo that needs no GPU: validates a payload, runs the dummy engine,
and prints the resulting metadata.

    python scripts/run_demo.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.schema import PromptConfig
from src.vision_core import DummyEngine, generate


def main() -> None:
    prompts = PromptConfig.from_file("configs/prompts.json")
    engine = DummyEngine({"paths": {"out_dir": tempfile.mkdtemp(prefix="flux_pulid_demo_")},
                          "standard": {"width": 512, "height": 512}})
    spec = prompts.prompts[0]
    result = generate({"prompt": spec.prompt, "ref_image": "reference.jpg",
                       "seed": prompts.main_seeds[0], "output_name": f"demo_{spec.prompt_id}",
                       "tags": {"prompt_id": spec.prompt_id, "style": spec.style}}, engine=engine)
    print(f"status={result.status}  image={result.image_path}")
    print(json.dumps({k: result.metadata[k] for k in ("engine", "width", "height", "ca_fired", "ca_expected")}, indent=2))


if __name__ == "__main__":
    main()
