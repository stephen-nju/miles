#!/usr/bin/env python3
"""One-click verification: is a chat template append-only after last user message?

Usage examples::

    # Verify a local .jinja template file
    python scripts/tools/verify_chat_template.py --template path/to/template.jinja

    # Verify a HuggingFace model's chat template
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B

    # Verify with autofix (use our fixed template if available)
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B --autofix

    # Restrict which append roles the session is allowed to use (tool is implicit)
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B \\
        --tito-allowed-append-roles user

    # Run thinking cases: off (default) / on / both
    python scripts/tools/verify_chat_template.py --model Qwen/Qwen3.5-0.8B --thinking both
"""

from __future__ import annotations

import argparse
import sys


def _load_template_from_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def _load_template_from_model(model_id: str, *, autofix: bool) -> tuple[str, str]:
    """Load chat template for a HF model. Returns (template_str, source_description)."""
    if autofix:
        from miles.utils.chat_template_utils.autofix import try_get_fixed_chat_template

        fixed_path = try_get_fixed_chat_template(model_id)
        if fixed_path:
            return _load_template_from_file(fixed_path), f"fixed template: {fixed_path}"

    from miles.utils.chat_template_utils.template import load_hf_chat_template

    return load_hf_chat_template(model_id), f"HuggingFace: {model_id}"


def _parse_chat_template_kwargs(pairs: list[str]) -> dict:
    """Parse ``KEY=VAL`` CLI args into a kwargs dict.

    ``true`` / ``false`` (case-insensitive) → Python bool; everything else is
    passed through as a string.  A template that genuinely needs the literal
    string ``"true"`` as a kwarg value has no current use case in this repo;
    add a JSON variant if that ever changes.
    """
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--chat-template-kwargs expected KEY=VAL; got {pair!r}")
        key, _, raw = pair.partition("=")
        lowered = raw.lower()
        if lowered == "true":
            out[key] = True
        elif lowered == "false":
            out[key] = False
        else:
            out[key] = raw
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a chat template is append-only after last user message.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--template",
        metavar="PATH",
        help="Path to a local .jinja chat template file.",
    )
    source.add_argument(
        "--model",
        metavar="MODEL_ID",
        help="HuggingFace model ID (e.g. Qwen/Qwen3-0.6B).",
    )

    parser.add_argument(
        "--autofix",
        action="store_true",
        help="When using --model, apply our fixed template if one exists.",
    )
    parser.add_argument(
        "--tito-allowed-append-roles",
        nargs="+",
        default=["tool"],
        choices=["tool", "user", "system"],
        metavar="ROLE",
        help=(
            "Roles the session may append after an assistant turn.  'tool' is "
            "implicitly always allowed (listing it is fine).  Trajectories that "
            "require roles outside this set are skipped.  Default: tool user system."
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=["off", "on", "both"],
        default="on",
        help=(
            "Thinking-mode filter.  off: non-thinking trajectories only.  "
            "on: thinking trajectories with enable_thinking=True.  "
            "both: non-thinking + thinking with enable_thinking={True,False}.  "
            "Default: off."
        ),
    )
    parser.add_argument(
        "--chat-template-kwargs",
        nargs="+",
        default=[],
        metavar="KEY=VAL",
        help=(
            "Extra kwargs threaded into the chat template on every render "
            "(e.g. 'clear_thinking=false' for GLM).  Values 'true'/'false' "
            "(case-insensitive) are parsed as bool; everything else is passed "
            "as a string."
        ),
    )

    args = parser.parse_args()

    extra_template_kwargs = _parse_chat_template_kwargs(args.chat_template_kwargs)

    # ── Load template ──────────────────────────────────────────────────
    if args.template:
        chat_template = _load_template_from_file(args.template)
        source_desc = f"file: {args.template}"
    else:
        chat_template, source_desc = _load_template_from_model(args.model, autofix=args.autofix)

    allowed_roles = set(args.tito_allowed_append_roles) | {"tool"}

    from miles.utils.test_utils.chat_template_verify import ALL_CASES, check_coverage, filter_cases, run_all_checks

    selected = filter_cases(ALL_CASES, allowed_append_roles=allowed_roles, thinking=args.thinking)

    print(f"Template source:       {source_desc}")
    print(f"Allowed append roles:  {sorted(allowed_roles)}")
    print(f"Thinking mode:         {args.thinking}")
    if extra_template_kwargs:
        print(f"Template kwargs:       {extra_template_kwargs}")
    print(f"Selected trajectories: {len(selected)} of {len(ALL_CASES)} (after filtering)")
    print()

    # Global coverage sanity check — reports gaps in the mock-trajectory pool,
    # not gaps caused by the current CLI flags.  A gap here means some CLI
    # setting exercises no trajectory; fixing it requires adding a trajectory
    # in mock_trajectories.py.
    coverage = check_coverage()
    if coverage.missing:
        print("Trajectory coverage gaps ((thinking, append_roles \\ {tool}) with no trajectory):")
        for is_thinking, roles in coverage.missing:
            label = "thinking    " if is_thinking else "non-thinking"
            roles_str = "{" + ", ".join(roles) + "}" if roles else "{}"
            print(f"  - {label}  x  {roles_str}")
        print()

    # ── Run verification ───────────────────────────────────────────────
    results = run_all_checks(
        chat_template,
        allowed_append_roles=allowed_roles,
        thinking=args.thinking,
        extra_template_kwargs=extra_template_kwargs,
    )

    # ── Print results ──────────────────────────────────────────────────
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    max_name_len = max((len(r.case_name) for r in results), default=0)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        line = f"  [{status}] {r.case_name:<{max_name_len}}"
        if r.error:
            first_line = r.error.split("\n")[0]
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            line += f"  -- {first_line}"
        print(line)

    print()
    print(f"Results: {passed}/{len(results)} passed, {failed} failed")

    if failed:
        print("\nVerdict: FAIL - template is NOT append-only after last user message")
        return 1
    else:
        print("\nVerdict: PASS - template IS append-only after last user message")
        return 0


if __name__ == "__main__":
    sys.exit(main())
