"""Reset GatedDeltaNet state at packed-document boundaries (FSDP backend).

The FSDP backend packs several documents into one forward (``--use-dynamic-batch-size``).
The stock-HF GatedDeltaNet runs its linear-attention recurrence (fla chunk/recurrent) and
its causal_conv1d over the whole packed row without sequence boundaries, so state bleeds
across documents and the train/rollout logprob diff inflates ~0.015 -> ~0.07. The
decoder-layer forward derives cu_seqlens/seq_idx from the packed position_ids and the
GatedDeltaNet forward feeds them to the kernels so both states reset per document.
GatedDeltaNet archs + THD packing (batch==1, >1 doc) only.
"""

import functools
import logging

logger = logging.getLogger(__name__)


def _cu_seqlens_from_position_ids(position_ids):
    import torch

    if position_ids is None or position_ids.dim() != 2 or position_ids.shape[0] != 1:
        return None
    pos = position_ids.reshape(-1)
    starts = (pos == 0).nonzero(as_tuple=True)[0]
    if starts.numel() <= 1:
        return None
    total = torch.tensor([pos.numel()], device=pos.device, dtype=starts.dtype)
    return torch.cat([starts, total]).to(torch.int32)


def _seq_idx_from_position_ids(position_ids):
    import torch

    if position_ids is None or position_ids.dim() != 2 or position_ids.shape[0] != 1:
        return None
    seq_idx = torch.cumsum((position_ids == 0).to(torch.int32), dim=-1) - 1
    if int(seq_idx.max()) <= 0:
        return None
    return seq_idx.to(torch.int32)


def _inject_kwarg(fn, key, value):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if kwargs.get(key) is None:
            kwargs[key] = value
        return fn(*args, **kwargs)

    return wrapped


# (kernel instance-attr, kwarg name, GatedDeltaNet attr holding the boundary tensor)
_INJECT = (
    ("chunk_gated_delta_rule", "cu_seqlens", "_gdn_cu_seqlens"),
    ("recurrent_gated_delta_rule", "cu_seqlens", "_gdn_cu_seqlens"),
    ("causal_conv1d_fn", "seq_idx", "_gdn_seq_idx"),
)


def _patch_gdn_forward(gdn_cls):
    orig = gdn_cls.forward
    if getattr(orig, "_gdn_packing", False):
        return

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        # The kernels are instance attributes; rebind them for this forward to inject the
        # per-document boundaries, then restore (nothing leaks across modules/forwards).
        saved = {}
        for attr, key, ctx_attr in _INJECT:
            value = getattr(self, ctx_attr, None)
            fn = getattr(self, attr, None)
            if fn is not None and value is not None:
                saved[attr] = fn
                setattr(self, attr, _inject_kwarg(fn, key, value))
        try:
            return orig(self, *args, **kwargs)
        finally:
            for attr, fn in saved.items():
                setattr(self, attr, fn)

    forward._gdn_packing = True
    gdn_cls.forward = forward


def _patch_decoder_forward(dl_cls, gdn_cls):
    orig = dl_cls.forward
    if getattr(orig, "_gdn_packing", False):
        return

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        pos = kwargs.get("position_ids")
        cu = _cu_seqlens_from_position_ids(pos)
        si = _seq_idx_from_position_ids(pos)
        if cu is not None or si is not None:
            for module in self.modules():
                if isinstance(module, gdn_cls):
                    module._gdn_cu_seqlens = cu
                    module._gdn_seq_idx = si
        return orig(self, *args, **kwargs)

    forward._gdn_packing = True
    dl_cls.forward = forward


def _find_class(mod, suffix):
    for name in dir(mod):
        if name.endswith(suffix):
            return getattr(mod, name)
    return None


def apply_gateddeltanet_packing_patch():
    """Idempotent; patches every GatedDeltaNet hybrid arch present (Qwen3.5/3.6, Qwen3-Next)."""
    patched = False
    for mod_name in ("qwen3_5_moe", "qwen3_next"):
        try:
            mod = __import__(f"transformers.models.{mod_name}.modeling_{mod_name}", fromlist=["x"])
        except Exception:
            continue
        gdn_cls = _find_class(mod, "GatedDeltaNet")
        dl_cls = _find_class(mod, "DecoderLayer")
        if gdn_cls is not None and dl_cls is not None:
            _patch_gdn_forward(gdn_cls)
            _patch_decoder_forward(dl_cls, gdn_cls)
            patched = True
    if patched:
        logger.info("[fsdp] GatedDeltaNet packing fix applied (cu_seqlens/seq_idx per packed doc)")
    return patched
