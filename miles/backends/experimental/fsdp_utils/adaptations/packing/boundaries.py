"""Shared packed-document boundary derivation for the FSDP backend.

The FSDP backend packs several documents into one ``[1, T]`` forward row (THD packing under
``--use-dynamic-batch-size``; with short responses many docs share a row). Stateful layers
(Mamba2 conv+scan, GatedDeltaNet linear-attention, attention) bleed across document boundaries
unless they reset per document. Every such arch needs the SAME information: where each packed
document starts. This module derives it once from the packed ``position_ids`` (which reset to 0
at each document start) so the per-arch packing specs share one definition instead of each
re-deriving cu_seqlens/seq_idx (the nemotron_h and qwen3_5_moe patches did so verbatim before).

``PackedSeqContext`` carries:
  * ``cu_seqlens`` ``[num_docs+1]`` int32 — varlen offsets (``flash_attn_varlen_func``, fla
    chunk/recurrent_gated_delta_rule)
  * ``seq_idx`` ``[1, T]`` int32 — per-token document id (``causal_conv1d_fn``,
    ``mamba_chunk_scan_combined``)
  * ``max_seqlen`` int — longest document (``flash_attn_varlen_func`` max_seqlen_q/k)

``packed_seq_context`` returns ``None`` for single-document / non-``[1, T]`` / batch>1 inputs —
i.e. the cases where packing is a no-op — so callers no-op exactly as the previous per-file
derivations did. A non-None context always has BOTH cu_seqlens and seq_idx set (they reset
together at the same document starts), matching the prior nemotron/qwen3.5 behavior.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackedSeqContext:
    cu_seqlens: "object"  # torch.Tensor int32 [num_docs+1]
    seq_idx: "object"  # torch.Tensor int32 [1, T]
    max_seqlen: int


def packed_seq_context(position_ids):
    """Derive per-document boundaries from packed ``position_ids``, or ``None`` when not packing."""
    import torch

    if position_ids is None or position_ids.dim() != 2 or position_ids.shape[0] != 1:
        return None
    pos = position_ids.reshape(-1)
    starts = (pos == 0).nonzero(as_tuple=True)[0]
    if starts.numel() <= 1:
        return None  # single document -> packing is a no-op
    total = torch.tensor([pos.numel()], device=pos.device, dtype=starts.dtype)
    cu_seqlens = torch.cat([starts, total]).to(torch.int32)
    seq_idx = (torch.cumsum((pos == 0).to(torch.int32), dim=0) - 1).to(torch.int32).unsqueeze(0).contiguous()
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
    return PackedSeqContext(cu_seqlens=cu_seqlens, seq_idx=seq_idx, max_seqlen=max_seqlen)
