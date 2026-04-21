"""Isolate whether Nemotron-H MoE training/rollout divergence lives in sglang's
forward or in Megatron-side weight loading / forward.

Strategy: on a fixed short token sequence, compute the model's per-token
log-probability P(tok_{i+1} | tok_0..tok_i) two ways:

  1. HF `transformers.NemotronHForCausalLM` — treated as gold reference.
  2. sglang offline `Engine(...)` with `return_logprob=True`.

Dump both to pickles, then a third mode diffs them token-by-token. If the
two disagree materially (> 1e-2 mean abs log-p diff) we have found the
ceiling-level culprit: it is sglang. Otherwise the culprit is on the
Megatron/miles side (training forward or weight conversion).

Usage:
  python scripts/tools/debug_nemotron_moe_logprob_diff.py hf     --model <path> --out hf.pkl
  python scripts/tools/debug_nemotron_moe_logprob_diff.py sglang --model <path> --out sg.pkl
  python scripts/tools/debug_nemotron_moe_logprob_diff.py diff   --hf hf.pkl --sglang sg.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import torch

PROMPT = (
    "The capital of France is"
)
# Keep prompt short — we care about per-token log-p agreement on the same
# fixed sequence, not about variety.

EXTRA_SEQ_LEN = 0  # number of additional tokens to probe beyond the prompt.


def _get_input_ids(model_path: str, seed: int = 0) -> list[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = tok(PROMPT, add_special_tokens=True, return_tensors="pt").input_ids[0].tolist()
    torch.manual_seed(seed)
    return ids


def run_hf(args):
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    print(f"[hf] loading {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[hf] loaded in {time.time() - t0:.1f}s", flush=True)

    input_ids = _get_input_ids(args.model)
    print(f"[hf] input_ids (len={len(input_ids)}): {input_ids}", flush=True)

    ids_t = torch.tensor([input_ids], device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(ids_t).logits  # [1, T, V]
    logits = logits.float().cpu()
    log_probs = torch.log_softmax(logits, dim=-1)  # [1, T, V]

    # Per-token log P(x_{t+1} | x_0..x_t): gather log_probs at position t
    # with index x_{t+1}.
    T = len(input_ids)
    per_token = []
    for t in range(T - 1):
        per_token.append(float(log_probs[0, t, input_ids[t + 1]]))

    # Also keep top-1 at each position so we can sanity-check agreement.
    top1 = logits[0].argmax(dim=-1).tolist()  # [T]

    out = {
        "mode": "hf",
        "model": args.model,
        "input_ids": input_ids,
        "per_token_logp": per_token,  # len T-1
        "top1_at_position": top1,
        "logits_sum_fp32": float(logits.sum()),
    }
    Path(args.out).write_bytes(pickle.dumps(out))
    print(f"[hf] wrote {args.out}", flush=True)
    print(f"[hf] per_token_logp = {per_token}", flush=True)
    print(f"[hf] top1_tokens = {top1}", flush=True)


def run_sglang(args):
    import os
    os.environ.setdefault("SGLANG_SKIP_NVIDIA_SMI_CHECK", "1")
    from sglang import Engine
    t0 = time.time()
    print(f"[sg] launching Engine tp={args.tp} on {args.model} ...", flush=True)
    engine = Engine(
        model_path=args.model,
        tp_size=args.tp,
        mem_fraction_static=0.7,
        trust_remote_code=True,
        disable_radix_cache=True,
        # Keep defaults otherwise so we exercise the same code path rollout uses.
    )
    print(f"[sg] engine up in {time.time() - t0:.1f}s", flush=True)

    input_ids = _get_input_ids(args.model)
    print(f"[sg] input_ids (len={len(input_ids)}): {input_ids}", flush=True)

    # Ask sglang for per-input-token logprobs. We don't care about generation,
    # so request 1 output token (minimum). `logprob_start_len=0` returns logprobs
    # for positions 1..T-1 (each P(x_t | x_0..x_{t-1})).
    result = engine.generate(
        input_ids=input_ids,
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
        top_logprobs_num=1,
    )
    print(f"[sg] raw keys: {list(result.keys())}", flush=True)
    meta = result.get("meta_info", {})
    print(f"[sg] meta keys: {list(meta.keys())}", flush=True)

    # `input_token_logprobs` is a list of (log_p, token_id, _) triples for
    # tokens 1..T-1. That is exactly what we want to compare to HF.
    it = meta.get("input_token_logprobs") or []
    print(f"[sg] input_token_logprobs len = {len(it)}; all: {it}", flush=True)
    # First entry is typically (None, first_token_id, None) — the first token
    # has no prior context. Keep only entries with a numeric log-p.
    per_token = [float(x[0]) for x in it if x[0] is not None]

    out = {
        "mode": "sglang",
        "model": args.model,
        "input_ids": input_ids,
        "per_token_logp": per_token,  # ideally len T-1, may be T-1 or T
        "raw_meta_first": dict(meta),  # dump meta for post-mortem
    }
    Path(args.out).write_bytes(pickle.dumps(out))
    print(f"[sg] wrote {args.out}", flush=True)
    print(f"[sg] per_token_logp = {per_token}", flush=True)

    engine.shutdown()


def run_diff(args):
    hf = pickle.loads(Path(args.hf).read_bytes())
    sg = pickle.loads(Path(args.sglang).read_bytes())
    print(f"[diff] hf  input_ids: {hf['input_ids']}")
    print(f"[diff] sg  input_ids: {sg['input_ids']}")
    assert hf["input_ids"] == sg["input_ids"], "input_ids differ — tokenization mismatch"

    hf_lp = hf["per_token_logp"]
    sg_lp = sg["per_token_logp"]
    # Align lengths — sglang may include position 0 or skip it.
    n = min(len(hf_lp), len(sg_lp))
    if len(hf_lp) != len(sg_lp):
        print(f"[diff] WARNING lengths differ hf={len(hf_lp)} sg={len(sg_lp)}; "
              f"comparing last {n}")
        hf_lp = hf_lp[-n:]
        sg_lp = sg_lp[-n:]
    diffs = [abs(a - b) for a, b in zip(hf_lp, sg_lp)]
    max_d = max(diffs) if diffs else 0.0
    mean_d = sum(diffs) / len(diffs) if diffs else 0.0
    print(f"[diff] N tokens compared: {n}")
    print(f"[diff] per-token diffs: {diffs}")
    print(f"[diff] max_abs_diff={max_d:.5f}  mean_abs_diff={mean_d:.5f}")
    if max_d < 1e-2:
        print("[diff] VERDICT: sglang and HF agree. Divergence source is NOT sglang; "
              "look at Megatron training forward / bridge weight load.")
    else:
        print("[diff] VERDICT: sglang diverges from HF. Fix sglang's nemotron_h MoE.")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_hf = sub.add_parser("hf")
    p_hf.add_argument("--model", required=True)
    p_hf.add_argument("--out", required=True)

    p_sg = sub.add_parser("sglang")
    p_sg.add_argument("--model", required=True)
    p_sg.add_argument("--out", required=True)
    p_sg.add_argument("--tp", type=int, default=4)

    p_d = sub.add_parser("diff")
    p_d.add_argument("--hf", required=True)
    p_d.add_argument("--sglang", required=True)

    args = p.parse_args()
    if args.cmd == "hf":
        run_hf(args)
    elif args.cmd == "sglang":
        run_sglang(args)
    else:
        run_diff(args)


if __name__ == "__main__":
    main()
