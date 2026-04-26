"""Verify that a chat template satisfies the append-only invariant.

The append-only invariant means: rendering the first N messages (without
generation prompt) produces a string that is an exact prefix of rendering
all messages (with generation prompt).  This is required by sglang's
pretokenized prefix mechanism for agentic workflows.

Core functions are used by both the CLI script
(``scripts/tools/verify_chat_template.py``) and the test suite
(``tests/fast/utils/chat_template_utils/test_pretokenized_chat.py``).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from miles.utils.chat_template_utils.template import apply_chat_template_from_str


def simulate_pretokenized_path(
    chat_template: str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    **template_kwargs,
) -> str:
    """Simulate the pretokenized incremental path at text level.

    1. Render first N messages (no generation prompt) -> prefix_text
    2. Render ALL messages (with generation prompt) -> full_text
    3. Verify prefix_text is a prefix of full_text

    Raises ``ValueError`` on prefix mismatch.
    """
    prefix_text = apply_chat_template_from_str(
        chat_template,
        messages[:pretokenized_num_message],
        add_generation_prompt=False,
        tools=tools,
        **template_kwargs,
    )

    full_text = apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tools,
        **template_kwargs,
    )

    if not full_text.startswith(prefix_text):
        raise ValueError(
            f"Prefix mismatch!\n"
            f"prefix_text ({len(prefix_text)} chars):\n{repr(prefix_text[-200:])}\n\n"
            f"full_text at same position:\n{repr(full_text[:len(prefix_text)][-200:])}"
        )

    return full_text


def get_standard_result(
    chat_template: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **template_kwargs,
) -> str:
    """Standard path: render all messages with generation prompt."""
    return apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tools,
        **template_kwargs,
    )


def assert_pretokenized_equals_standard(chat_template, messages, pretokenized_num_message, tools=None, **kwargs):
    """Assert pretokenized incremental path produces same text as standard full render."""
    standard = get_standard_result(chat_template, messages, tools=tools, **kwargs)
    pretokenized = simulate_pretokenized_path(chat_template, messages, pretokenized_num_message, tools=tools, **kwargs)
    assert pretokenized == standard, f"Pretokenized (N={pretokenized_num_message}) != standard"


# ---------------------------------------------------------------------------
# Non-raising verification API for CLI / programmatic use
# ---------------------------------------------------------------------------


@dataclass
class VerifyResult:
    """Result of a single append-only verification case."""

    case_name: str
    passed: bool
    error: str | None = None


def verify_append_only(
    chat_template: str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    case_name: str = "",
    **template_kwargs,
) -> VerifyResult:
    """Check that the template satisfies the append-only invariant.

    Returns a ``VerifyResult`` instead of raising, making it suitable for
    batch verification in CLI scripts.
    """
    try:
        standard = get_standard_result(chat_template, deepcopy(messages), tools=tools, **template_kwargs)
        pretokenized = simulate_pretokenized_path(
            chat_template, deepcopy(messages), pretokenized_num_message, tools=tools, **template_kwargs
        )
        if pretokenized != standard:
            return VerifyResult(
                case_name=case_name, passed=False, error=f"Pretokenized (N={pretokenized_num_message}) != standard"
            )
        return VerifyResult(case_name=case_name, passed=True)
    except ValueError as e:
        return VerifyResult(case_name=case_name, passed=False, error=str(e))
    except Exception as e:
        return VerifyResult(case_name=case_name, passed=False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Built-in test cases (shared between CLI and test suite)
# ---------------------------------------------------------------------------
#
# Trajectories expose two class attributes used for verify-layer filtering:
#
#   * ``APPEND_ROLES: frozenset[str]`` — non-assistant roles that appear after
#     the first assistant message.  Drives ``--tito-allowed-append-roles``.
#   * ``IS_THINKING: bool`` — any assistant carries ``reasoning_content``.
#     Drives ``--thinking`` and whether ``enable_thinking`` kwarg is passed.
#
# Both are declared on the trajectory class (mock_trajectories.py), alongside
# ``TOOLS`` / ``PRETOKENIZE_POSITIONS`` / ``MESSAGES``.  This file only lists
# which trajectories to exercise and expands them into concrete cases.

import re  # noqa: E402

from miles.utils.test_utils.mock_trajectories import (  # noqa: E402
    IntermediateSystemThinkingTrajectory,
    IntermediateSystemTrajectory,
    LongChainThinkingTrajectory,
    LongChainTrajectory,
    MultiToolSingleTurnTrajectory,
    MultiTurnNoToolThinkingTrajectory,
    MultiTurnNoToolTrajectory,
    MultiTurnThinkingTrajectory,
    MultiTurnTrajectory,
    MultiUserToolChainTrajectory,
    MultiUserTurnThinkingTrajectory,
    ParallelToolsTrajectory,
    RetrySystemTrajectory,
    SimpleNoToolTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)


def _short_name(cls: type) -> str:
    name = cls.__name__.replace("Trajectory", "")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# Trajectories exercised by ``run_all_checks`` / the CLI.  Must be a subset of
# the classes defined in mock_trajectories.py.  Callers (CLI, tests) pick
# the applicable subset via ``filter_cases`` / ``expand_runs`` based on
# each template's supported append roles and thinking mode; there is no
# global "exclude" list here.
_TRAJECTORIES: list[type] = [
    SingleToolTrajectory,
    MultiTurnTrajectory,
    MultiToolSingleTurnTrajectory,
    ParallelToolsTrajectory,
    LongChainTrajectory,
    MultiUserToolChainTrajectory,
    RetrySystemTrajectory,
    IntermediateSystemTrajectory,
    SimpleNoToolTrajectory,
    MultiTurnNoToolTrajectory,
    SingleToolThinkingTrajectory,
    MultiTurnThinkingTrajectory,
    LongChainThinkingTrajectory,
    MultiUserTurnThinkingTrajectory,
    IntermediateSystemThinkingTrajectory,
    MultiTurnNoToolThinkingTrajectory,
]


@dataclass(frozen=True)
class CaseSpec:
    """One verify case with classification metadata copied from its trajectory."""

    case_name: str
    traj_cls: type
    pretokenize_n: int
    tools: list[dict] | None
    append_roles: frozenset[str]
    is_thinking: bool


def _expand(traj_cls: type) -> list[CaseSpec]:
    """Expand one trajectory into one CaseSpec per PRETOKENIZE_POSITIONS value."""
    short = _short_name(traj_cls)
    return [
        CaseSpec(
            case_name=f"{short}-N{n}",
            traj_cls=traj_cls,
            pretokenize_n=n,
            tools=traj_cls.TOOLS,
            append_roles=traj_cls.APPEND_ROLES,
            is_thinking=traj_cls.IS_THINKING,
        )
        for n in traj_cls.PRETOKENIZE_POSITIONS
    ]


ALL_CASES: list[CaseSpec] = [c for t in _TRAJECTORIES for c in _expand(t)]

# ``tool`` is always considered allowed when filtering: the session layer
# assumes a tool-capable agent, so ``--tito-allowed-append-roles`` only needs
# to specify the *optional* roles (user, system).
_IMPLICIT_ALLOWED_ROLES: frozenset[str] = frozenset({"tool"})

THINKING_MODES: tuple[str, ...] = ("off", "on", "both")


def filter_cases(
    cases: list[CaseSpec],
    *,
    allowed_append_roles: set[str] | frozenset[str],
    thinking: str,
) -> list[CaseSpec]:
    """Select cases whose append roles fit *allowed_append_roles* and whose
    thinking flag fits *thinking* (``"off"`` / ``"on"`` / ``"both"``).

    ``tool`` is unioned into *allowed_append_roles* automatically.  ``both``
    accepts either thinking value; the actual expansion into
    ``enable_thinking=True/False`` happens in :func:`run_all_checks`.
    """
    if thinking not in THINKING_MODES:
        raise ValueError(f"thinking must be one of {THINKING_MODES}; got {thinking!r}")

    allowed = frozenset(allowed_append_roles) | _IMPLICIT_ALLOWED_ROLES
    out: list[CaseSpec] = []
    for c in cases:
        if not c.append_roles.issubset(allowed):
            continue
        if thinking == "off" and c.is_thinking:
            continue
        if thinking == "on" and not c.is_thinking:
            continue
        out.append(c)
    return out


def enable_thinking_variants(thinking: str) -> list[dict]:
    """Return the list of ``enable_thinking`` kwarg variants to apply per case.

    * ``"off"`` → ``[{}]`` (no ``enable_thinking`` kwarg).
    * ``"on"``  → ``[{"enable_thinking": True}]``.
    * ``"both"`` → ``[{"enable_thinking": True}, {"enable_thinking": False}]``.

    Both CLI (:func:`run_all_checks`) and pytest test-id expansion
    (:func:`expand_runs`) use this to avoid drifting in how the
    ``enable_thinking`` knob is exercised.
    """
    if thinking == "off":
        return [{}]
    if thinking == "on":
        return [{"enable_thinking": True}]
    if thinking == "both":
        return [{"enable_thinking": True}, {"enable_thinking": False}]
    raise ValueError(f"thinking must be one of {THINKING_MODES}; got {thinking!r}")


def format_case_id(case: CaseSpec, kwargs: dict) -> str:
    """Human-readable label for a ``(case, template_kwargs)`` tuple.

    Used for both CLI ``VerifyResult.case_name`` and pytest test ids so the
    same tuple is identified the same way in both surfaces.  Format:

    * empty kwargs → ``case.case_name``.
    * otherwise → ``<case.case_name>-<k1>_on/off-<k2>=val`` (keys sorted;
      bool values emit ``key_on`` / ``key_off``; other values ``key=val``).
    """
    if not kwargs:
        return case.case_name
    parts: list[str] = []
    for k, v in sorted(kwargs.items()):
        if isinstance(v, bool):
            parts.append(f"{k}_{'on' if v else 'off'}")
        else:
            parts.append(f"{k}={v}")
    return f"{case.case_name}-{'-'.join(parts)}"


def expand_runs(
    *,
    supports_thinking: bool,
    allowed_append_roles: frozenset[str] | None = None,
    extra_template_kwargs: dict | None = None,
):
    """Yield ``(case, template_kwargs)`` pairs consistent with a template's
    capability, for use by pytest parametrize in template / alignment tests.

    * ``supports_thinking``: False filters out thinking cases and yields no
      ``enable_thinking`` kwarg.  True yields both ``enable_thinking=True``
      and ``enable_thinking=False`` for every selected case (thinking or
      not), so the template's thinking branch is exercised against
      non-reasoning input too.
    * ``allowed_append_roles``: ``None`` means all roles; otherwise cases
      whose ``append_roles`` is not a subset are skipped.  ``tool`` is
      unioned in implicitly (matches :func:`filter_cases`).
    * ``extra_template_kwargs``: merged into every yielded kwargs dict —
      used to thread template-specific kwargs like GLM's
      ``clear_thinking=False``.

    Thin shell over :func:`filter_cases` + :func:`enable_thinking_variants`;
    the CLI path (:func:`run_all_checks`) uses the same helpers so that the
    expansion semantics cannot drift between pytest and CLI.
    """
    thinking = "both" if supports_thinking else "off"
    roles = allowed_append_roles if allowed_append_roles is not None else frozenset({"tool", "user", "system"})
    extra = extra_template_kwargs or {}
    variants = enable_thinking_variants(thinking)
    for c in filter_cases(ALL_CASES, allowed_append_roles=roles, thinking=thinking):
        for variant in variants:
            yield c, {**variant, **extra}


@dataclass
class CoverageReport:
    """Coverage of cases across ``(is_thinking, append_roles \\ {tool})``.

    ``covered`` maps each combination to the case names that fall in it;
    ``missing`` lists combinations with no case.  ``tool`` is excluded from
    the role axis because it is implicitly always allowed.
    """

    covered: dict[tuple[bool, tuple[str, ...]], list[str]]
    missing: list[tuple[bool, tuple[str, ...]]]


def check_coverage(
    cases: list[CaseSpec] | None = None,
    *,
    role_universe: set[str] | None = None,
) -> CoverageReport:
    """Enumerate ``thinking × append-role-subset`` combinations and report gaps.

    Used as a sanity check that every meaningful combination of
    ``--tito-allowed-append-roles`` and ``--thinking`` is backed by at least
    one trajectory — otherwise certain CLI settings would be no-ops.
    """
    if cases is None:
        cases = ALL_CASES
    if role_universe is None:
        role_universe = {"user", "system"}

    from itertools import chain, combinations

    ordered_universe = sorted(role_universe)
    all_subsets: list[tuple[str, ...]] = [
        tuple(sub)
        for sub in chain.from_iterable(combinations(ordered_universe, r) for r in range(len(ordered_universe) + 1))
    ]

    covered: dict[tuple[bool, tuple[str, ...]], list[str]] = {
        (is_thinking, sub): [] for is_thinking in (False, True) for sub in all_subsets
    }
    for c in cases:
        roles_key = tuple(sorted(c.append_roles - _IMPLICIT_ALLOWED_ROLES))
        key = (c.is_thinking, roles_key)
        if key in covered:
            covered[key].append(c.case_name)

    missing = [k for k, v in covered.items() if not v]
    return CoverageReport(covered=covered, missing=missing)


def run_all_checks(
    chat_template: str,
    *,
    allowed_append_roles: set[str] | frozenset[str] | None = None,
    thinking: str = "off",
    extra_template_kwargs: dict | None = None,
) -> list[VerifyResult]:
    """Run verification cases filtered by *allowed_append_roles* and *thinking*.

    ``allowed_append_roles`` is the set of roles the session may append after
    an assistant turn.  ``tool`` is implicit; defaults to
    ``{"tool", "user", "system"}`` (all roles, matching the pre-refactor
    behavior of ``include_intermediate_system=True``).  Trajectories whose
    required roles are not a subset of the allow list are skipped.

    ``thinking`` selects which ``enable_thinking`` variants are exercised —
    see :func:`enable_thinking_variants`.  When ``"both"``, **every** selected
    trajectory (thinking or not) is rerun with ``enable_thinking=True`` and
    ``enable_thinking=False``, so templates that branch on the flag are
    validated against non-reasoning input too.

    ``extra_template_kwargs`` is merged into every invocation — use it to
    thread template-specific kwargs (e.g. GLM's ``clear_thinking=False``)
    through the CLI.
    """
    if allowed_append_roles is None:
        allowed_append_roles = {"tool", "user", "system"}
    extra = extra_template_kwargs or {}

    selected = filter_cases(ALL_CASES, allowed_append_roles=allowed_append_roles, thinking=thinking)
    variants = enable_thinking_variants(thinking)

    results: list[VerifyResult] = []
    for case in selected:
        for variant in variants:
            kwargs = {**variant, **extra}
            results.append(
                verify_append_only(
                    chat_template,
                    deepcopy(case.traj_cls.MESSAGES),
                    case.pretokenize_n,
                    tools=case.tools,
                    case_name=format_case_id(case, kwargs),
                    **kwargs,
                )
            )

    return results
