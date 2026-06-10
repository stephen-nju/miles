from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_PARENT = _WORKSPACE_ROOT.parent
_LOCAL_IFEVAL_REQUIREMENTS = _WORKSPACE_ROOT / "examples" / "eval_multi_task" / "requirements_ifeval.txt"

# The original IFEval ships inside the multi-GB google-research monorepo, so we
# sparse-checkout only the instruction_following_eval/ package instead of cloning
# the whole tree. This is the sibling of ifbench.py, which targets AllenAI's
# extended IFBench; here we target the original 25-instruction Google benchmark.
_IFEVAL_CHECKOUT = _WORKSPACE_PARENT / "google-research-ifeval"
_GOOGLE_RESEARCH_REPO = "https://github.com/google-research/google-research.git"
_IFEVAL_PACKAGE = "instruction_following_eval"


JsonDict = dict[str, Any]
KwargsDict = dict[str, str | int | float | None]

# Cached official module; loaded lazily on first scoring (see _get_evaluation_lib).
_evaluation_lib = None


def _ensure_ifeval_repo() -> Path:
    """Sparse-checkout the official instruction_following_eval package and expose it
    on sys.path.

    We put the checkout *root* (not the package dir) on the path and later import
    the package-qualified ``instruction_following_eval.evaluation_lib``. Two reasons:
    the package's own modules import each other as ``from instruction_following_eval
    import ...``, and the qualified name avoids clashing with IFBench, whose fork
    ships an identically named top-level ``evaluation_lib`` module."""

    package_path = _IFEVAL_CHECKOUT / _IFEVAL_PACKAGE
    if not package_path.exists():
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    _GOOGLE_RESEARCH_REPO,
                    str(_IFEVAL_CHECKOUT),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(_IFEVAL_CHECKOUT), "sparse-checkout", "set", _IFEVAL_PACKAGE],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(_IFEVAL_CHECKOUT), "checkout"],
                check=True,
                capture_output=True,
            )
        except Exception as exc:
            raise ImportError(
                "Unable to automatically fetch the official Google IFEval source. "
                "Clone https://github.com/google-research/google-research.git and "
                f"sparse-checkout '{_IFEVAL_PACKAGE}' into {_IFEVAL_CHECKOUT}."
            ) from exc

    root = str(_IFEVAL_CHECKOUT)
    if root not in sys.path:
        sys.path.insert(0, root)

    current_pythonpath = os.environ.get("PYTHONPATH")
    if current_pythonpath is None:
        os.environ["PYTHONPATH"] = root
    elif root not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([root, current_pythonpath])

    return _IFEVAL_CHECKOUT


def _ensure_ifeval_dependencies() -> None:
    """Install IFEval requirements the first time the lib fails to import."""

    requirements_file = _LOCAL_IFEVAL_REQUIREMENTS

    if not requirements_file.exists():
        logger.debug("Local IFEval requirements file not found at %s; skipping install.", requirements_file)
        return

    sentinel = _IFEVAL_CHECKOUT / ".deps_installed"
    if sentinel.exists():
        return

    install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
    try:
        subprocess.run(install_cmd, check=True)
    except Exception as exc:
        logger.warning("Failed to install IFEval dependencies automatically: %s", exc)
    else:
        sentinel.write_text("installed\n")


def _load_evaluation_lib():
    _ensure_ifeval_repo()
    try:
        return importlib.import_module(f"{_IFEVAL_PACKAGE}.evaluation_lib")
    except ImportError:
        _ensure_ifeval_dependencies()
        return importlib.import_module(f"{_IFEVAL_PACKAGE}.evaluation_lib")


def _get_evaluation_lib():
    """Return the official evaluation_lib, loading it lazily on first use.

    Loading clones the source and may pip-install, so we defer it until a reward is
    actually computed. Importing this module (for the pure helpers, or under unit
    tests that stub the lib) must stay free of that network/IO side effect."""

    global _evaluation_lib
    if _evaluation_lib is None:
        _evaluation_lib = _load_evaluation_lib()
    return _evaluation_lib


def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> list[str]:
    """Ensure instruction identifiers are clean strings."""

    normalized: list[str] = []
    for entry in raw_ids or []:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _coerce_kwargs_list(raw_kwargs: Any, num_instructions: int) -> list[KwargsDict]:
    """Convert stored kwargs into the per-instruction list IFEval expects.

    None values are dropped so that ``build_description(**kwargs)`` only receives the
    keys a given instruction actually declares; the dataset stores every possible
    kwarg key per row with unused ones set to null."""

    if isinstance(raw_kwargs, list):
        processed: list[KwargsDict] = []
        for entry in raw_kwargs:
            if isinstance(entry, dict):
                processed.append(dict(entry))
            else:
                processed.append({})
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    else:
        processed = [{} for _ in range(num_instructions)]

    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    sanitized: list[KwargsDict] = []
    for entry in processed:
        sanitized.append({k: v for k, v in entry.items() if v is not None})
    return sanitized


def _build_input_example(metadata: JsonDict, input_example_cls):
    """Build the official InputExample from stored metadata.

    The InputExample class is injected rather than imported at module level so the
    lib stays lazily loaded and unit tests can pass a stand-in."""

    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list") or [])
    if not instruction_ids:
        logger.debug("Missing instruction identifiers in metadata: %s", metadata)
        return None

    prompt_text = metadata.get("prompt_text")
    prompt_text = "" if prompt_text is None else str(prompt_text)

    kwargs_list = _coerce_kwargs_list(metadata.get("kwargs"), len(instruction_ids))

    return input_example_cls(
        key=int(metadata.get("record_id") or 0),
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )


def compute_ifeval_reward(
    response: str, label: Any, metadata: JsonDict | None = None, *, strict: bool = True
) -> float:
    """Score a model response using the official Google IFEval rules.

    Sibling of compute_ifbench_reward but for the original 25-instruction IFEval.
    ``strict`` selects between the two official criteria, exposed as the
    ``ifeval_strict`` / ``ifeval_loose`` reward types: strict checks the raw
    response, while loose retries the check against response variants with the
    leading/trailing line or surrounding ``*`` markdown stripped."""

    if metadata is None:
        logger.debug("No metadata provided for IFEval scoring.")
        return 0.0

    if response is None:
        return 0.0

    evaluation_lib = _get_evaluation_lib()
    inp = _build_input_example(metadata, evaluation_lib.InputExample)
    if inp is None:
        return 0.0

    prompt_to_response = {inp.prompt: str(response or "")}
    evaluate = (
        evaluation_lib.test_instruction_following_strict if strict else evaluation_lib.test_instruction_following_loose
    )
    output = evaluate(inp, prompt_to_response)
    return 1.0 if output.follow_all_instructions else 0.0
