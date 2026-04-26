from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest
from transformers import AutoTokenizer

from miles.utils.chat_template_utils import MismatchType, apply_chat_template, resolve_fixed_chat_template
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, TITOTokenizerType, get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.mock_trajectories import SingleToolTrajectory

_ALLOWED_APPEND_ROLES = ["tool"]
_TOK_CACHE: dict[tuple[str, str | None], AutoTokenizer] = {}


@dataclass(frozen=True)
class TITOModelCase:
    model_name: str
    tito_type: TITOTokenizerType
    chat_template_path: str | None = None
    assistant_start_str: str | None = None


_TITO_MODEL_CASES = [
    TITOModelCase(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        tito_type=TITOTokenizerType.DEFAULT,
        assistant_start_str="<|im_start|>assistant\n",
    ),
    TITOModelCase(
        model_name="Qwen/Qwen3-0.6B",
        tito_type=TITOTokenizerType.QWEN3,
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWEN3, ["tool"]),
    ),
    TITOModelCase(
        model_name="Qwen/Qwen3.5-0.8B",
        tito_type=TITOTokenizerType.QWEN35,
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWEN35, ["tool"]),
    ),
    TITOModelCase(
        model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",
        tito_type=TITOTokenizerType.QWENNEXT,
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWENNEXT, ["tool"]),
    ),
    TITOModelCase(
        model_name="zai-org/GLM-4.7-Flash",
        tito_type=TITOTokenizerType.GLM47,
    ),
]


@dataclass(frozen=True)
class AppendCase:
    name: str
    old_messages: list[dict]
    appended_messages: list[dict]
    tools: list[dict] | None
    required_contents: tuple[str, ...] = ()


def _single_tool_case() -> AppendCase:
    return AppendCase(
        name="single_tool",
        old_messages=deepcopy(SingleToolTrajectory.MESSAGES[:3]),
        appended_messages=deepcopy([SingleToolTrajectory.MESSAGES[3]]),
        tools=deepcopy(SingleToolTrajectory.TOOLS),
        required_contents=(SingleToolTrajectory.MESSAGES[3]["content"],),
    )


_SINGLE_APPEND_CASES = [_single_tool_case()]


def _iter_single_append_params() -> list[pytest.ParamSpecArgs]:
    params = []
    for model_case in _TITO_MODEL_CASES:
        for case in _SINGLE_APPEND_CASES:
            params.append(pytest.param(model_case, case, id=f"{case.name}-{model_case.model_name}"))
    return params


_SINGLE_APPEND_PARAMS = _iter_single_append_params()


def _get_tokenizer(model_case: TITOModelCase) -> AutoTokenizer:
    cache_key = (model_case.model_name, model_case.chat_template_path)
    if cache_key not in _TOK_CACHE:
        _TOK_CACHE[cache_key] = load_tokenizer(
            model_case.model_name,
            chat_template_path=model_case.chat_template_path,
            trust_remote_code=True,
        )
    return _TOK_CACHE[cache_key]


def _get_tito(model_case: TITOModelCase, tokenizer: AutoTokenizer) -> TITOTokenizer:
    kwargs = {
        "tokenizer_type": model_case.tito_type,
        "allowed_append_roles": _ALLOWED_APPEND_ROLES,
    }
    if model_case.tito_type == TITOTokenizerType.DEFAULT:
        kwargs["assistant_start_str"] = model_case.assistant_start_str
    return get_tito_tokenizer(tokenizer, **kwargs)


def _render_ids(
    tokenizer: AutoTokenizer, messages: list[dict], tools: list[dict] | None, *, add_generation_prompt: bool
) -> list[int]:
    return apply_chat_template(
        messages,
        tokenizer=tokenizer,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
    )


def _assert_only_assistant_mismatches(tito: TITOTokenizer, expected: list[int], merged: list[int]) -> None:
    mismatches = tito.create_comparator().compare_sequences(expected, merged)
    bad = [m for m in mismatches if m.type != MismatchType.ASSISTANT_TEXT]
    assert not bad, [m.to_dict() for m in bad]


def _assert_contents_in_order(
    incremental_text: str, required_contents: tuple[str, ...], *, model_case: TITOModelCase, case_name: str
) -> None:
    cursor = 0
    for content in required_contents:
        found = incremental_text.find(content, cursor)
        assert found >= 0, f"model_name={model_case.model_name!r} {case_name=} missing ordered content {content!r}"
        cursor = found + len(content)


def _run_case(model_case: TITOModelCase, case: AppendCase) -> tuple[TITOTokenizer, list[int], list[int], str]:
    tokenizer = _get_tokenizer(model_case)
    tito = _get_tito(model_case, tokenizer)
    old_messages = deepcopy(case.old_messages)
    new_messages = old_messages + deepcopy(case.appended_messages)
    try:
        expected = _render_ids(tokenizer, new_messages, case.tools, add_generation_prompt=True)
        pretokenized = _render_ids(tokenizer, old_messages, case.tools, add_generation_prompt=False)
    except Exception as exc:
        pytest.skip(f"{model_case.model_name} cannot render case {case.name}: {type(exc).__name__}: {exc}")
    merged = tito.merge_tokens(old_messages, new_messages, pretokenized, case.tools)
    incremental_text = tokenizer.decode(tito.tokenize_additional_non_assistant(old_messages, new_messages, case.tools))
    return tito, merged, expected, incremental_text


@pytest.mark.parametrize(("model_case", "case"), _SINGLE_APPEND_PARAMS)
def test_single_append_cases_preserve_non_assistant_content(model_case: TITOModelCase, case: AppendCase):
    tito, merged, expected, incremental_text = _run_case(model_case, case)
    _assert_only_assistant_mismatches(tito, expected, merged)
    _assert_contents_in_order(incremental_text, case.required_contents, model_case=model_case, case_name=case.name)
