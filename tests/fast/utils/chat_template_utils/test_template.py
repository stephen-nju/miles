"""Test that apply_chat_template aligns with SGLang's _apply_jinja_template.

The reference function ``sglang_prompt_ids`` calls
``OpenAIServingChat._process_messages`` directly — the *actual* SGLang code
path, not a re-implementation.  A lightweight ``OpenAIServingChat`` instance
is constructed via ``object.__new__`` (bypassing ``__init__``) with only the
attributes that ``_process_messages`` / ``_apply_jinja_template`` read:

- ``tokenizer_manager.tokenizer`` — the HF tokenizer under test
- ``template_manager.chat_template_name = None`` → selects the Jinja path
- ``template_manager.jinja_template_content_format = "string"`` → text-only
- ``use_dpsk_v32_encoding = False`` / ``is_gpt_oss = False``

Each test asserts that our ``apply_chat_template`` produces identical token IDs.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from transformers import AutoTokenizer

from miles.utils.chat_template_utils.autofix import try_get_fixed_chat_template
from miles.utils.chat_template_utils.template import apply_chat_template
from miles.utils.test_utils.chat_template_verify import CaseSpec, expand_runs, format_case_id
from miles.utils.test_utils.mock_trajectories import SimpleNoToolTrajectory, SingleToolTrajectory

# ---------------------------------------------------------------------------
# SGLang reference: calls OpenAIServingChat._process_messages directly
# ---------------------------------------------------------------------------


def _make_serving(tokenizer) -> OpenAIServingChat:
    """Create a minimal ``OpenAIServingChat`` that can run ``_process_messages``."""
    serving = object.__new__(OpenAIServingChat)
    serving.tokenizer_manager = MagicMock()
    serving.tokenizer_manager.tokenizer = tokenizer
    serving.template_manager = MagicMock()
    serving.template_manager.chat_template_name = None
    serving.template_manager.jinja_template_content_format = "string"
    serving.use_dpsk_v32_encoding = False
    serving.is_gpt_oss = False
    serving.tool_call_parser = None
    serving.reasoning_parser = None
    return serving


def sglang_prompt_ids(
    tokenizer,
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs,
) -> list[int]:
    """Get prompt_ids by calling SGLang's ``_process_messages`` directly."""
    request_data: dict = {"messages": copy.deepcopy(messages), "model": "test"}
    if tools:
        request_data["tools"] = copy.deepcopy(tools)
    if kwargs:
        request_data["chat_template_kwargs"] = kwargs
    request = ChatCompletionRequest(**request_data)

    serving = _make_serving(tokenizer)
    result = serving._process_messages(request, is_multimodal=False)
    return result.prompt_ids


# ---------------------------------------------------------------------------
# Tokenizer cache & fixed-template loader
# ---------------------------------------------------------------------------

_TOK_CACHE: dict[str, AutoTokenizer] = {}


def _get_tokenizer(model_id: str) -> AutoTokenizer:
    if model_id not in _TOK_CACHE:
        _TOK_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return _TOK_CACHE[model_id]


def _load_fixed_or_none(hf_id: str) -> str | None:
    """Return the bundled fixed chat-template content for *hf_id*, or ``None``."""
    path = try_get_fixed_chat_template(hf_id)
    if path is None:
        return None
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Per-model declarations
# ---------------------------------------------------------------------------
#
# (model_id, supports_thinking, use_fixed_template, allowed_append_roles)
#
# ``allowed_append_roles`` reflects the set of append-role combinations the
# model's template can render without raising — test asserts that the
# sglang path and our path produce identical tokens on all such cases.
# Qwen3.5-4B uses the bundled fixed template which raises on intermediate
# system post-revert, so the role set is narrowed to {tool} only.

_MODELS: list[tuple[str, bool, bool, frozenset[str]]] = [
    ("Qwen/Qwen3-4B", True, False, frozenset({"tool", "user", "system"})),
    ("zai-org/GLM-4.7-Flash", True, False, frozenset({"tool", "user", "system"})),
    ("Qwen/Qwen3.5-4B", True, True, frozenset({"tool"})),
    ("Qwen/Qwen3-Coder-Next", False, False, frozenset({"tool", "user", "system"})),
]


def _build_align_params():
    params = []
    for model_id, supports_thinking, use_fixed, allowed_roles in _MODELS:
        short = model_id.split("/")[-1]
        for case, kwargs in expand_runs(supports_thinking=supports_thinking, allowed_append_roles=allowed_roles):
            # SimpleNoTool ends with a plain assistant message (no tool_calls).
            # sglang's _process_messages treats that as continue_final_message
            # and drops the trailing <|im_start|>assistant header, which
            # diverges from apply_chat_template with add_generation_prompt=True.
            # Out of scope for the alignment test.
            if case.traj_cls is SimpleNoToolTrajectory:
                continue
            ident = f"{short}-{format_case_id(case, kwargs)}"
            params.append(pytest.param(model_id, use_fixed, case, kwargs, id=ident))
    return params


def _per_model_params():
    return [pytest.param(model_id, use_fixed, id=model_id.split("/")[-1]) for model_id, _, use_fixed, _ in _MODELS]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _assert_aligned(tokenizer, case: CaseSpec, kwargs: dict, chat_template: str | None):
    # ``kwargs`` are template kwargs (e.g. ``enable_thinking``, ``clear_thinking``)
    # produced by ``expand_runs`` — both ``sglang_prompt_ids`` (via
    # ``chat_template_kwargs``) and ``apply_chat_template`` (via ``**template_kwargs``)
    # route them into jinja.  Don't put non-template kwargs in here.
    extra = {"chat_template": chat_template} if chat_template else {}
    expected = sglang_prompt_ids(tokenizer, case.traj_cls.MESSAGES, case.traj_cls.TOOLS, **kwargs, **extra)
    actual = apply_chat_template(
        case.traj_cls.MESSAGES,
        tokenizer=tokenizer,
        tools=case.traj_cls.TOOLS,
        tokenize=True,
        **kwargs,
        **extra,
    )
    assert actual == expected


class TestAlignWithSGLang:
    """apply_chat_template must produce identical prompt_ids to SGLang's pipeline."""

    @pytest.mark.parametrize("model_id, use_fixed, case, kwargs", _build_align_params())
    def test_align(self, model_id, use_fixed, case, kwargs):
        tokenizer = _get_tokenizer(model_id)
        chat_template = _load_fixed_or_none(model_id) if use_fixed else None
        _assert_aligned(tokenizer, case, kwargs, chat_template)

    @pytest.mark.parametrize("model_id, use_fixed", _per_model_params())
    def test_json_string_arguments(self, model_id, use_fixed):
        """JSON-string tool_call arguments should produce same IDs as dict arguments."""
        tokenizer = _get_tokenizer(model_id)
        chat_template = _load_fixed_or_none(model_id) if use_fixed else None
        extra = {"chat_template": chat_template} if chat_template else {}
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                    }
                ],
            },
            {"role": "tool", "content": "sunny", "tool_call_id": "call_1", "name": "get_weather"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        expected = sglang_prompt_ids(tokenizer, messages, tools, **extra)
        actual = apply_chat_template(messages, tokenizer=tokenizer, tools=tools, tokenize=True, **extra)
        assert actual == expected

    @pytest.mark.parametrize("model_id, use_fixed", _per_model_params())
    def test_no_tools(self, model_id, use_fixed):
        """Plain conversation without tools."""
        tokenizer = _get_tokenizer(model_id)
        chat_template = _load_fixed_or_none(model_id) if use_fixed else None
        extra = {"chat_template": chat_template} if chat_template else {}
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        expected = sglang_prompt_ids(tokenizer, messages, **extra)
        actual = apply_chat_template(messages, tokenizer=tokenizer, tokenize=True, **extra)
        assert actual == expected

    @pytest.mark.parametrize("model_id, use_fixed", _per_model_params())
    def test_does_not_mutate_input(self, model_id, use_fixed):
        tokenizer = _get_tokenizer(model_id)
        messages = copy.deepcopy(SingleToolTrajectory.MESSAGES)
        tools = copy.deepcopy(SingleToolTrajectory.TOOLS)
        saved_msgs = copy.deepcopy(messages)
        saved_tools = copy.deepcopy(tools)
        apply_chat_template(messages, tokenizer=tokenizer, tools=tools, tokenize=True)
        assert messages == saved_msgs
        assert tools == saved_tools
