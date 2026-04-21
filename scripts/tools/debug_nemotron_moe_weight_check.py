"""Verify AutoBridge→Megatron weight load matches HF expert tensors bit-exactly.

Runs single-rank (TP=1 PP=1 EP=1) so every expert lives locally. Picks a few
Mamba-H MoE layers, loads the expert weights two ways:

  1. Directly from HF safetensors — this is the gold reference.
  2. Via `AutoBridge.to_megatron_provider(load_weights=True)` then introspects
     the resulting Megatron `MambaModel` state_dict.

Diffs corresponding tensors. A non-zero diff points straight at the bridge
mapping (expert orientation / concat / routing).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/tools/debug_nemotron_moe_weight_check.py \
        --model /cluster_public/miles_data/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def _setup_dist():
    import datetime

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        backend="nccl", rank=0, world_size=1, timeout=datetime.timedelta(minutes=30)
    )
    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )

    # Mcore requires model-parallel-rng to be added before any param init.
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(42)


def _load_hf_experts(model_path: str, layer_idx: int, expert_ids: list[int]) -> dict[str, torch.Tensor]:
    """Read expert tensors directly from HF safetensors.

    Returns a dict keyed by HF param name (e.g.
    `backbone.layers.{layer_idx}.mixer.experts.{i}.up_proj.weight`).
    """
    from safetensors import safe_open

    # The layer may store experts across multiple shards; scan all.
    target_names = set()
    for eid in expert_ids:
        target_names.add(f"backbone.layers.{layer_idx}.mixer.experts.{eid}.up_proj.weight")
        target_names.add(f"backbone.layers.{layer_idx}.mixer.experts.{eid}.down_proj.weight")
    target_names.add(f"backbone.layers.{layer_idx}.mixer.gate.weight")
    target_names.add(f"backbone.layers.{layer_idx}.mixer.gate.e_score_correction_bias")
    target_names.add(f"backbone.layers.{layer_idx}.mixer.shared_experts.up_proj.weight")
    target_names.add(f"backbone.layers.{layer_idx}.mixer.shared_experts.down_proj.weight")

    shards = sorted(Path(model_path).glob("*.safetensors"))
    found: dict[str, torch.Tensor] = {}
    for s in shards:
        with safe_open(s, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in target_names:
                    found[k] = f.get_tensor(k)
    return found


def _find_megatron_expert_tensors(model, layer_idx: int) -> dict[str, torch.Tensor]:
    """Extract the relevant MoE parameters from the Megatron MambaModel for one layer.

    Returns a flat dict keyed by local Megatron param name, e.g.
    `decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight0`.
    """
    prefix = f"decoder.layers.{layer_idx}.mlp"
    collected: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if name.startswith(prefix):
            collected[name] = p.detach().cpu()
        # Also grab router buffers / expert_bias which are non-param buffers.
    for name, b in model.named_buffers():
        if name.startswith(prefix):
            collected[name] = b.detach().cpu()
    return collected


def run(args):
    print(f"[setup] init single-rank distributed ...", flush=True)
    _setup_dist()

    # Force imports that patch the bridge with miles' MoE extension shim.
    import miles.backends.megatron_utils  # noqa: F401

    from megatron.bridge import AutoBridge

    t0 = time.time()
    print(f"[bridge] AutoBridge.from_hf_pretrained({args.model}) ...", flush=True)
    bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = False
    provider.context_parallel_size = 1
    # MoE settings (scoring, scaling_factor, etc.) come from the miles shim in
    # miles/backends/megatron_utils/__init__.py — it pulls them from hf_config.
    # Small seqlen just to instantiate cheaply.
    provider.seq_length = 64
    provider.attention_softmax_in_fp32 = True
    provider.finalize()
    print(f"[bridge] provider ready in {time.time() - t0:.1f}s; "
          f"num_moe_experts={provider.num_moe_experts} "
          f"moe_ffn_hidden_size={provider.moe_ffn_hidden_size} "
          f"moe_layer_freq-head={(provider.moe_layer_freq or [])[:10]}", flush=True)

    model = provider.provide(pre_process=True, post_process=True, vp_stage=None)
    print(f"[bridge] model provided in {time.time() - t0:.1f}s; loading HF weights ...", flush=True)
    bridge.load_hf_weights(model)
    print(f"[bridge] weights loaded in {time.time() - t0:.1f}s", flush=True)

    # Pick a known MoE layer: first 'E' in the hybrid pattern of Nano-30B is
    # position 1 (pattern starts 'MEMEM*'). We'll grab a few MoE layers.
    pat = provider.hybrid_override_pattern
    moe_layer_indices = [i for i, c in enumerate(pat) if c == "E"]
    print(f"[bridge] MoE layer indices: {moe_layer_indices[:10]} ... total={len(moe_layer_indices)}", flush=True)

    for layer_idx in moe_layer_indices[:args.num_layers]:
        expert_ids = [0, 1, 63, 64, 127]  # sample across the range
        print(f"\n===== layer {layer_idx} =====", flush=True)

        hf_tensors = _load_hf_experts(args.model, layer_idx, expert_ids)
        mg_tensors = _find_megatron_expert_tensors(model, layer_idx)
        print(f"[check] hf keys: {sorted(hf_tensors.keys())}", flush=True)
        print(f"[check] mg keys: {sorted(mg_tensors.keys())}", flush=True)

        # Gate / router weight
        hf_gate = hf_tensors.get(f"backbone.layers.{layer_idx}.mixer.gate.weight")
        mg_gate_candidates = [k for k in mg_tensors if "router.weight" in k]
        if hf_gate is not None and mg_gate_candidates:
            mg_gate = mg_tensors[mg_gate_candidates[0]]
            d = (hf_gate.float() - mg_gate.float()).abs().max().item()
            print(f"[gate]  max_abs_diff={d:.6e}  shape hf={tuple(hf_gate.shape)} mg={tuple(mg_gate.shape)}", flush=True)

        # Expert bias
        hf_bias = hf_tensors.get(f"backbone.layers.{layer_idx}.mixer.gate.e_score_correction_bias")
        mg_bias_candidates = [k for k in mg_tensors if "expert_bias" in k]
        if hf_bias is not None and mg_bias_candidates:
            mg_bias = mg_tensors[mg_bias_candidates[0]]
            d = (hf_bias.float() - mg_bias.float()).abs().max().item()
            print(f"[bias]  max_abs_diff={d:.6e}  shape hf={tuple(hf_bias.shape)} mg={tuple(mg_bias.shape)}", flush=True)

        # Expert up_proj / down_proj — compare per-expert.
        for eid in expert_ids:
            hf_up = hf_tensors.get(f"backbone.layers.{layer_idx}.mixer.experts.{eid}.up_proj.weight")
            hf_down = hf_tensors.get(f"backbone.layers.{layer_idx}.mixer.experts.{eid}.down_proj.weight")
            # Megatron TE-grouped stores each as linear_fc1.weight{eid}, linear_fc2.weight{eid}.
            mg_up = mg_tensors.get(f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight{eid}")
            mg_down = mg_tensors.get(f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2.weight{eid}")
            if mg_up is None or mg_down is None:
                # Older naming with local_experts.{eid}.linear_fc1.weight
                mg_up = mg_up or mg_tensors.get(
                    f"decoder.layers.{layer_idx}.mlp.experts.local_experts.{eid}.linear_fc1.weight"
                )
                mg_down = mg_down or mg_tensors.get(
                    f"decoder.layers.{layer_idx}.mlp.experts.local_experts.{eid}.linear_fc2.weight"
                )
            if hf_up is not None and mg_up is not None:
                # Orient check: HF up_proj is [ffn_hidden, hidden]. Megatron fc1 may be
                # [ffn_hidden, hidden] (row-major weight) or [hidden, ffn_hidden].
                hu, mu = hf_up.float(), mg_up.float()
                shape_match = hu.shape == mu.shape
                shape_tpose = hu.shape == tuple(reversed(mu.shape))
                if shape_match:
                    d = (hu - mu).abs().max().item()
                elif shape_tpose:
                    d = (hu - mu.t()).abs().max().item()
                else:
                    d = float("nan")
                print(f"[e{eid}.up]   hf={tuple(hu.shape)} mg={tuple(mu.shape)} match={shape_match} tpose={shape_tpose} max_abs_diff={d:.6e}", flush=True)
            if hf_down is not None and mg_down is not None:
                hd, md = hf_down.float(), mg_down.float()
                shape_match = hd.shape == md.shape
                shape_tpose = hd.shape == tuple(reversed(md.shape))
                if shape_match:
                    d = (hd - md).abs().max().item()
                elif shape_tpose:
                    d = (hd - md.t()).abs().max().item()
                else:
                    d = float("nan")
                print(f"[e{eid}.down] hf={tuple(hd.shape)} mg={tuple(md.shape)} match={shape_match} tpose={shape_tpose} max_abs_diff={d:.6e}", flush=True)

        # (forward check happens once after all layer spot-checks)
        # Shared experts
        hf_su = hf_tensors.get(f"backbone.layers.{layer_idx}.mixer.shared_experts.up_proj.weight")
        hf_sd = hf_tensors.get(f"backbone.layers.{layer_idx}.mixer.shared_experts.down_proj.weight")
        mg_su = mg_tensors.get(f"decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc1.weight")
        mg_sd = mg_tensors.get(f"decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc2.weight")
        for label, h, m in (("shared.up", hf_su, mg_su), ("shared.down", hf_sd, mg_sd)):
            if h is not None and m is not None:
                hu, mu = h.float(), m.float()
                if hu.shape == mu.shape:
                    d = (hu - mu).abs().max().item()
                    note = "direct"
                elif hu.shape == tuple(reversed(mu.shape)):
                    d = (hu - mu.t()).abs().max().item()
                    note = "tpose"
                else:
                    d = float("nan")
                    note = f"shape-mismatch hf={tuple(hu.shape)} mg={tuple(mu.shape)}"
                print(f"[{label}] max_abs_diff={d:.6e} ({note})", flush=True)

    # After spot-checking a few layers' weights, run a full forward pass and
    # compare logprobs to HF.
    _forward_check(model, args)


def _forward_check(model, args):
    """Run a forward pass on a fixed 5-token sequence and compare logprobs to HF dump.

    Reads the HF pickle from args.hf_logp (produced by debug_nemotron_moe_logprob_diff.py).
    Reports per-token logprob diff Megatron-vs-HF. This is the decisive test: if the
    weight match is bit-exact AND forward logprobs disagree, the bug is in Megatron's
    MoE forward (routing_replay / grouped_gemm / activation / scaling).
    """
    import pickle
    from pathlib import Path

    if not args.hf_logp:
        print("[fwd] --hf-logp not provided; skipping forward check", flush=True)
        return

    hf = pickle.loads(Path(args.hf_logp).read_bytes())
    input_ids = hf["input_ids"]
    hf_per_token = hf["per_token_logp"]
    print(f"[fwd] input_ids={input_ids}  hf_per_token={hf_per_token}", flush=True)

    device = torch.cuda.current_device()
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    T = ids.shape[1]
    position_ids = torch.arange(T, device=device).unsqueeze(0)
    # Causal mask (standard Megatron: `attention_mask=None` triggers causal).

    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=ids,
            position_ids=position_ids,
            attention_mask=None,
        )
    # Megatron MambaModel returns logits [B, T, V] when post_process=True.
    logits = out.float().cpu()
    print(f"[fwd] logits shape={tuple(logits.shape)}", flush=True)
    log_probs = torch.log_softmax(logits, dim=-1)

    mg_per_token = [float(log_probs[0, t, input_ids[t + 1]]) for t in range(T - 1)]
    top1 = logits[0].argmax(dim=-1).tolist()
    print(f"[fwd] megatron top1={top1}", flush=True)
    print(f"[fwd] megatron per_token_logp={mg_per_token}", flush=True)
    print(f"[fwd] hf       per_token_logp={hf_per_token}", flush=True)
    diffs = [abs(a - b) for a, b in zip(hf_per_token, mg_per_token)]
    max_d = max(diffs) if diffs else float("nan")
    mean_d = sum(diffs) / len(diffs) if diffs else float("nan")
    print(f"[fwd] VERDICT per-token diffs={diffs} max={max_d:.5f} mean={mean_d:.5f}", flush=True)
    if max_d < 1e-2:
        print("[fwd] Megatron forward matches HF — bug must be elsewhere (rollout / capture).", flush=True)
    else:
        print("[fwd] Megatron forward DIVERGES from HF — the MoE training forward is the bug.", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--num-layers", type=int, default=3,
                   help="Number of MoE layers to check (from the start).")
    p.add_argument("--hf-logp", type=str, default=None,
                   help="Path to HF logprob pkl produced by debug_nemotron_moe_logprob_diff.py hf")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
