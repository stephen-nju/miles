"""
Unit tests for the pretokenized chat completion path.

Tests that using pretokenized_token_ids + pretokenized_num_message produces
identical token IDs as the standard apply_chat_template path.

Ported from sglang test/unit/test_pretokenized_chat.py.
"""

from copy import deepcopy

import pytest

from miles.utils.chat_template_utils.autofix import try_get_fixed_chat_template
from miles.utils.chat_template_utils.template import load_hf_chat_template
from miles.utils.test_utils.chat_template_verify import (
    CaseSpec,
    assert_pretokenized_equals_standard,
    expand_runs,
    format_case_id,
    simulate_pretokenized_path,
)
from miles.utils.test_utils.mock_trajectories import (
    MultiTurnTrajectory,
    MultiUserTurnThinkingTrajectory,
    SingleToolTrajectory,
    last_user_index,
)


def _load_fixed(hf_id: str) -> str:
    path = try_get_fixed_chat_template(hf_id)
    assert path is not None, f"try_get_fixed_chat_template should resolve {hf_id}"
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Per-template capability declarations
# ---------------------------------------------------------------------------
#
# Each entry: (name, content, supports_thinking, allowed_append_roles, extra_template_kwargs)
#
# Design:
# * fixed templates only test {tool} — fixed series is designed for tool-calling
#   agents; user / system append are explicitly out of scope.
# * GLM thinking templates are covered twice: a default entry (tool only, no
#   kwargs) matching production behavior, and a *_clear_thinking_off entry that
#   adds {user} to allowed roles with clear_thinking=False to preserve
#   reasoning across user turns (GLM's canonical pattern for session reset).
# * Other non-thinking HF native templates only test {tool} — scope aligned
#   with fixed.
# * system role is never tested here; those trajectories remain in ALL_CASES
#   for CLI use but are filtered out of this test by the above role sets.

_TEMPLATES: list[tuple[str, str, bool, frozenset[str], dict]] = [
    # fixed templates: tool only
    ("qwen3_fixed", _load_fixed("Qwen/Qwen3-0.6B"), True, frozenset({"tool"}), {}),
    ("qwen3.5_fixed", _load_fixed("Qwen/Qwen3.5-0.8B"), True, frozenset({"tool"}), {}),
    ("qwen3_thinking_2507_fixed", _load_fixed("Qwen/Qwen3-4B-Thinking-2507"), True, frozenset({"tool"}), {}),
    ("qwen3_next_thinking_fixed", _load_fixed("Qwen/Qwen3-Next-80B-A3B-Thinking"), True, frozenset({"tool"}), {}),
    # GLM thinking: default (tool only) + user-append variant with clear_thinking=False
    ("glm5", load_hf_chat_template("zai-org/GLM-5"), True, frozenset({"tool"}), {}),
    (
        "glm5_clear_thinking_off",
        load_hf_chat_template("zai-org/GLM-5"),
        True,
        frozenset({"tool", "user"}),
        {"clear_thinking": False},
    ),
    ("glm47_flash", load_hf_chat_template("zai-org/GLM-4.7-Flash"), True, frozenset({"tool"}), {}),
    (
        "glm47_flash_clear_thinking_off",
        load_hf_chat_template("zai-org/GLM-4.7-Flash"),
        True,
        frozenset({"tool", "user"}),
        {"clear_thinking": False},
    ),
    # other HF native non-thinking: tool only
    ("qwen3_instruct_2507", load_hf_chat_template("Qwen/Qwen3-4B-Instruct-2507"), False, frozenset({"tool"}), {}),
    ("qwen3_next_instruct", load_hf_chat_template("Qwen/Qwen3-Next-80B-A3B-Instruct"), False, frozenset({"tool"}), {}),
    ("qwen3_coder_next", load_hf_chat_template("Qwen/Qwen3-Coder-Next"), False, frozenset({"tool"}), {}),
    ("glm4", load_hf_chat_template("THUDM/glm-4-9b-chat"), False, frozenset({"tool"}), {}),
]


# Original (unfixed) HF templates referenced by negative tests
_ORIGINAL_TEMPLATES = {
    "qwen3_original": load_hf_chat_template("Qwen/Qwen3-0.6B"),
    "qwen3_thinking_2507": load_hf_chat_template("Qwen/Qwen3-4B-Thinking-2507"),
    "qwen3_next_thinking": load_hf_chat_template("Qwen/Qwen3-Next-80B-A3B-Thinking"),
}


def _build_pretokenized_params():
    params = []
    for name, content, supports_thinking, allowed_roles, extra_kwargs in _TEMPLATES:
        for case, kwargs in expand_runs(
            supports_thinking=supports_thinking,
            allowed_append_roles=allowed_roles,
            extra_template_kwargs=extra_kwargs,
        ):
            ident = f"{name}-{format_case_id(case, kwargs)}"
            params.append(pytest.param(content, case, kwargs, id=ident))
    return params


# ===========================================================================
# Core tests: every (template, case, kwargs) tuple satisfies append-only
# ===========================================================================


@pytest.mark.parametrize("chat_template, case, kwargs", _build_pretokenized_params())
def test_pretokenized(chat_template: str, case: CaseSpec, kwargs: dict):
    assert_pretokenized_equals_standard(
        chat_template=chat_template,
        messages=deepcopy(case.traj_cls.MESSAGES),
        pretokenized_num_message=case.pretokenize_n,
        tools=case.tools,
        **kwargs,
    )


# ===========================================================================
# Negative tests: original (unfixed) templates fail prefix invariant
# ===========================================================================


# (chat_template, trajectory_cls, pretokenize_n)
_MISMATCH_CASES = [
    pytest.param(_ORIGINAL_TEMPLATES["qwen3_original"], SingleToolTrajectory, 3, id="qwen3_original-single_tool"),
    pytest.param(_ORIGINAL_TEMPLATES["qwen3_original"], MultiTurnTrajectory, 3, id="qwen3_original-multi_turn"),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_thinking_2507"], SingleToolTrajectory, 3, id="qwen3_thinking_2507-single_tool"
    ),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_next_thinking"], SingleToolTrajectory, 3, id="qwen3_next_thinking-single_tool"
    ),
    pytest.param(
        _ORIGINAL_TEMPLATES["qwen3_next_thinking"], MultiTurnTrajectory, 3, id="qwen3_next_thinking-multi_turn"
    ),
]


@pytest.mark.parametrize("chat_template,trajectory_cls,pretokenize_n", _MISMATCH_CASES)
def test_original_template_prefix_mismatch(chat_template, trajectory_cls, pretokenize_n):
    """Original templates with loop.last cause prefix mismatch (our fix resolves this)."""
    with pytest.raises(ValueError, match="Prefix mismatch"):
        simulate_pretokenized_path(
            chat_template,
            deepcopy(trajectory_cls.MESSAGES),
            pretokenize_n,
            tools=trajectory_cls.TOOLS,
        )


# ===========================================================================
# Negative test: cross-user-turn thinking compression breaks prefix invariant
# ===========================================================================

# Pretokenizing BEFORE the last user turn in a multi-user-turn thinking
# trajectory fails because templates compress reasoning_content from earlier
# turns.  This is a known template limitation, not a bug in the fixed templates.
_CROSS_USER_THINKING_N = last_user_index(MultiUserTurnThinkingTrajectory.MESSAGES)


def _unique_thinking_templates():
    seen: set[str] = set()
    out = []
    for name, content, supports_thinking, _, _ in _TEMPLATES:
        if not supports_thinking:
            continue
        if content in seen:
            continue
        seen.add(content)
        out.append(pytest.param(content, id=name))
    return out


@pytest.mark.parametrize("chat_template", _unique_thinking_templates())
@pytest.mark.parametrize("enable_thinking", [True, False], ids=["thinking_on", "thinking_off"])
def test_cross_user_turn_thinking_prefix_mismatch(chat_template, enable_thinking):
    """Thinking templates compress reasoning_content from earlier user turns, breaking prefix invariant."""
    with pytest.raises(ValueError, match="Prefix mismatch"):
        simulate_pretokenized_path(
            chat_template,
            deepcopy(MultiUserTurnThinkingTrajectory.MESSAGES),
            _CROSS_USER_THINKING_N,
            tools=MultiUserTurnThinkingTrajectory.TOOLS,
            enable_thinking=enable_thinking,
        )
