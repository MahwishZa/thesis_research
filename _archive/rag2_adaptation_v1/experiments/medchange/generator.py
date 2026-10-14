"""The local generator (llama.cpp, CPU) and the configuration recorded next to every file of model output.

The generator itself (model file hash, context size, token limit, seed, ...) is recorded once per output file in
``<output>.config.json``. A resumed run whose result-relevant settings differ from the recorded ones is refused, so one
output file never mixes configurations. Used by every command that runs a model: ``generate_answers``, ``rag2_run``,
``label_audit`` and ``verdict_consistency``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

from . import arms as A
from .prompts import SYSTEM, TEMPLATE

SEED = 42
#: Settings whose change makes two answers incomparable. ``n_threads``, ``n_gpu_layers`` and the
#: llama-cpp-python version are recorded for the record but do not block a resume.
RESULT_RELEVANT = ("model_sha256", "n_ctx", "max_new_tokens", "temperature", "seed",
                   "arm_settings", "system_sha256", "template_sha256")


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_config(*, model_path, model_sha256: str, n_ctx: int, max_new_tokens: int,
               n_threads: Optional[int], n_gpu_layers: int, llama_version: str) -> dict:
    """Everything about the generator that an answers file needs to be reproduced."""
    return {"model_file": Path(model_path).name, "model_sha256": model_sha256,
            "n_ctx": n_ctx, "max_new_tokens": max_new_tokens, "temperature": 0.0, "seed": SEED,
            "arm_settings": A.settings_hash(), "system_sha256": _text_sha256(SYSTEM),
            "template_sha256": _text_sha256(TEMPLATE),
            "n_threads": n_threads, "n_gpu_layers": n_gpu_layers,
            "llama_cpp_python": llama_version}


def config_path(answers: Path) -> Path:
    return answers.with_name(answers.stem + ".config.json")


def check_config(answers: Path, config: dict, relevant=RESULT_RELEVANT) -> list[str]:
    """Bind an output file to the configuration that produced it.

    Records ``config`` next to ``answers`` when nothing is recorded yet and returns the
    ``relevant`` fields on which it disagrees with an existing record (empty: compatible).
    """
    path = config_path(answers)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
        return []
    recorded = json.loads(path.read_text(encoding="utf-8"))
    return [k for k in relevant if recorded.get(k) != config.get(k)]


def llama_version() -> str:
    try:
        import llama_cpp
    except ImportError:
        return "not installed"
    return getattr(llama_cpp, "__version__", "unknown")


def llama_generator(model_path: str, n_ctx: int, n_threads: Optional[int], n_gpu_layers: int,
                    max_new_tokens: int) -> Callable[[str, str], str]:
    from llama_cpp import Llama
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                n_gpu_layers=n_gpu_layers, verbose=False, seed=SEED)

    def generate(system: str, user: str) -> str:
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_new_tokens, temperature=0.0)
        return r["choices"][0]["message"]["content"]
    return generate
