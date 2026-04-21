import logging

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLMoETextRotaryEmbedding,
        Qwen3VLTextRotaryEmbedding,
    )

    def patch_rotary_embedding(cls):
        _original_forward = cls.forward

        def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
            return _original_forward(self, *args, **kwargs)

        cls.forward = _patched_forward

    patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
    patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
except ImportError:
    pass

logging.getLogger("megatron").setLevel(logging.WARNING)


try:
    # nemotron_h (Mamba+Attention hybrid) surfaces >2-tuple at
    # TransformerLayer._forward_attention's unpack. Three-layer defense:
    # (1) SelfAttention.forward (if that's where the extra element appears)
    # (2) SelfAttention.__call__ (covers nn.Module dispatch + hooks)
    # (3) Diagnostic print of actual self_attention class and tuple size
    #     inside TransformerLayer._forward_attention so we can see what's
    #     happening when all else fails.
    import sys as _miles_sys
    _miles_sys.stderr.write(">>> miles nemotron_h attn-shim: installing\n")
    _miles_sys.stderr.flush()

    from megatron.core.transformer.attention import SelfAttention as _MilesSelfAttention
    from megatron.core.transformer.transformer_layer import TransformerLayer as _MilesTL

    _miles_attn_diag_logged = [False]

    _orig_sa_forward = _MilesSelfAttention.forward

    def _miles_sa_forward(self, *args, **kwargs):
        ret = _orig_sa_forward(self, *args, **kwargs)
        if isinstance(ret, tuple) and len(ret) > 2:
            return ret[0], ret[1]
        return ret

    _MilesSelfAttention.forward = _miles_sa_forward

    _orig_sa_call = _MilesSelfAttention.__call__

    def _miles_sa_call(self, *args, **kwargs):
        ret = _orig_sa_call(self, *args, **kwargs)
        if isinstance(ret, tuple) and len(ret) > 2:
            return ret[0], ret[1]
        return ret

    _MilesSelfAttention.__call__ = _miles_sa_call

    _orig_fwd_attn = _MilesTL._forward_attention

    def _miles_fwd_attn(self, *args, **kwargs):
        # nemotron_h / Mamba hybrid: non-attention positions have IdentityOp
        # in self_attention. TransformerLayer._forward_attention's unpack
        # fails on the raw tensor returned. Short-circuit.
        if type(self.self_attention).__name__ == "IdentityOp":
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            return hidden_states, None
        return _orig_fwd_attn(self, *args, **kwargs)

    _MilesTL._forward_attention = _miles_fwd_attn

    _orig_fwd_mlp = _MilesTL._forward_mlp

    def _miles_fwd_mlp(self, hidden_states, *args, **kwargs):
        # Symmetric fix: attention-only layers have IdentityOp in self.mlp.
        # _forward_mlp's `mlp_output, mlp_output_bias = mlp_output_with_bias`
        # unpack fails on IdentityOp return (raw tensor). Short-circuit so
        # the attention output flows through unchanged.
        if type(self.mlp).__name__ == "IdentityOp":
            return hidden_states
        return _orig_fwd_mlp(self, hidden_states, *args, **kwargs)

    _MilesTL._forward_mlp = _miles_fwd_mlp

    # For PP>1: megatron.bridge's param_mapping.broadcast_obj_from_pp_rank
    # calls torch.distributed.broadcast_object_list with the pp_group.
    # miles wraps groups in ReloadableProcessGroup (subclass of
    # torch.distributed.ProcessGroup) which is NOT registered in
    # torch.distributed._world.pg_group_ranks, so get_group_rank raises
    # "Group ... is not registered". Unwrap to the inner real group before
    # the broadcast.
    from miles.utils.reloadable_process_group import ReloadableProcessGroup as _MilesRPG
    from megatron.bridge.models.conversion import param_mapping as _MilesBridgeParamMapping

    _orig_broadcast_obj_from_pp_rank = _MilesBridgeParamMapping.MegatronParamMapping.broadcast_obj_from_pp_rank

    def _miles_broadcast_obj_from_pp_rank(self, obj, name=None):
        if isinstance(self.pp_group, _MilesRPG):
            _orig_pp = self.pp_group
            self.pp_group = _orig_pp.group  # inner real torch ProcessGroup
            try:
                return _orig_broadcast_obj_from_pp_rank(self, obj, name)
            finally:
                self.pp_group = _orig_pp
        return _orig_broadcast_obj_from_pp_rank(self, obj, name)

    _MilesBridgeParamMapping.MegatronParamMapping.broadcast_obj_from_pp_rank = _miles_broadcast_obj_from_pp_rank

    # Extend upstream NemotronHBridge for MoE variants (e.g. Nemotron-3-Nano-30B-A3B,
    # Super-120B-A12B). The upstream bridge only maps dense Mamba+Attention weights;
    # MoE checkpoints add `mixer.gate.*`, `mixer.experts.*`, `mixer.shared_experts.*`
    # under the HF `backbone.layers.*.mixer.*` namespace, and the provider needs
    # num_moe_experts / moe_router_* fields.
    from megatron.bridge.models.nemotronh.nemotron_h_bridge import NemotronHBridge as _MilesNHBridge
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping as _MilesAutoMapping,
    )
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry as _MilesMappingRegistry,
    )

    _orig_nh_provider_bridge = _MilesNHBridge.provider_bridge
    _orig_nh_mapping_registry = _MilesNHBridge.mapping_registry

    def _miles_nh_provider_bridge(self, hf_pretrained):
        provider = _orig_nh_provider_bridge(self, hf_pretrained)
        hf = hf_pretrained.config
        # Nemotron-H uses DeepSeek-style config naming for MoE (n_routed_experts).
        n_exp = (
            getattr(hf, "num_experts", None)
            or getattr(hf, "n_routed_experts", None)
            or 0
        )
        # Stash HF config on the bridge so mapping_registry() can reach it
        # (upstream base class only stores hf_pretrained on the parent AutoBridge).
        self._miles_hf_config = hf
        self._miles_num_experts = int(n_exp)
        if n_exp > 0:
            provider.num_moe_experts = int(n_exp)
            provider.moe_router_topk = int(hf.num_experts_per_tok)
            # For Mamba-hybrid, `hybrid_override_pattern` (e.g. "MEMEM*...")
            # drives which positions are MoE. Mirror to `moe_layer_freq` so
            # miles' replay_utils.py can detect MoE layers correctly.
            pat = getattr(provider, "hybrid_override_pattern", None) or getattr(
                hf, "hybrid_override_pattern", None
            )
            if pat:
                provider.moe_layer_freq = [1 if ch == "E" else 0 for ch in pat][
                    : int(provider.num_layers)
                ]
            provider.moe_router_score_function = "sigmoid"
            provider.moe_router_enable_expert_bias = True
            provider.moe_grouped_gemm = True
            provider.moe_ffn_hidden_size = int(
                getattr(hf, "moe_intermediate_size", None) or provider.ffn_hidden_size
            )
            shared_size = getattr(hf, "moe_shared_expert_intermediate_size", None)
            if shared_size is not None:
                provider.moe_shared_expert_intermediate_size = int(shared_size)
            elif getattr(hf, "n_shared_experts", 0):
                provider.moe_shared_expert_intermediate_size = (
                    int(provider.moe_ffn_hidden_size) * int(hf.n_shared_experts)
                )
            # Pull routing hyperparameters directly from HF config. miles' generic
            # model_provider.py only copies a couple of moe_* args to the provider,
            # so without this Megatron runs with scaling_factor=None which silently
            # makes every MoE block 2.5x smaller than HF (0.28 logprob drift at
            # rollout time).
            rsf = getattr(hf, "routed_scaling_factor", None)
            if rsf is not None:
                provider.moe_router_topk_scaling_factor = float(rsf)
            n_group = getattr(hf, "n_group", None)
            if n_group is not None:
                provider.moe_router_num_groups = int(n_group)
            topk_group = getattr(hf, "topk_group", None)
            if topk_group is not None:
                provider.moe_router_group_topk = int(topk_group)
            if getattr(hf, "norm_topk_prob", None) is not None:
                # Megatron's sigmoid path normalizes topk_prob when topk>1;
                # there's no separate flag. Leave as-is but record for clarity.
                pass
            provider.moe_router_dtype = "fp32"
            # pre_softmax is a softmax-path flag; irrelevant for sigmoid routing.
            # Upstream miles model_provider already passes --moe-router-bias-update-rate
            # and --moe-aux-loss-coeff through to the provider.
        return provider

    def _miles_nh_mapping_registry(self):
        # Always append MoE mappings. For dense nemotron_h models the extra
        # megatron params (mlp.router.*, mlp.experts.*, mlp.shared_experts.*)
        # won't exist, so these mappings are unreferenced and harmless. For
        # MoE variants they enable round-trip HF↔Megatron conversion.
        registry = _orig_nh_mapping_registry(self)

        extra_mappings = {
            # Router
            "decoder.layers.*.mlp.router.weight": "backbone.layers.*.mixer.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": (
                "backbone.layers.*.mixer.gate.e_score_correction_bias"
            ),
            # Routed experts (up-only FFN — nemotron_h uses squared_relu, no gate)
            "decoder.layers.*.mlp.experts.linear_fc1.weight*": (
                "backbone.layers.*.mixer.experts.*.up_proj.weight"
            ),
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": (
                "backbone.layers.*.mixer.experts.*.down_proj.weight"
            ),
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight": (
                "backbone.layers.*.mixer.experts.*.up_proj.weight"
            ),
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": (
                "backbone.layers.*.mixer.experts.*.down_proj.weight"
            ),
            # Shared experts
            "decoder.layers.*.mlp.shared_experts.linear_fc1.weight": (
                "backbone.layers.*.mixer.shared_experts.up_proj.weight"
            ),
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": (
                "backbone.layers.*.mixer.shared_experts.down_proj.weight"
            ),
        }

        new_mappings = list(registry.mappings) if hasattr(registry, "mappings") else list(registry._mappings)  # type: ignore
        for mg, hf_key in extra_mappings.items():
            new_mappings.append(_MilesAutoMapping(megatron_param=mg, hf_param=hf_key))
        return _MilesMappingRegistry(*new_mappings)

    _MilesNHBridge.provider_bridge = _miles_nh_provider_bridge
    _MilesNHBridge.mapping_registry = _miles_nh_mapping_registry

    # One-shot expert_bias magnitude check on first router forward.
    try:
        from megatron.core.transformer.moe.router import TopKRouter as _MilesRouter
        _orig_router_fwd = _MilesRouter.forward
        _miles_router_logged = [False]
        def _miles_router_fwd(self, *args, **kwargs):
            if (not _miles_router_logged[0]) and hasattr(self, 'expert_bias') and self.expert_bias is not None:
                eb = self.expert_bias
                _miles_sys.stderr.write(
                    f">>> miles router-diag: expert_bias abs.max={eb.abs().max().item():.4f} "
                    f"abs.mean={eb.abs().mean().item():.4f} shape={tuple(eb.shape)} dtype={eb.dtype}\n"
                )
                _miles_sys.stderr.flush()
                _miles_router_logged[0] = True
            return _orig_router_fwd(self, *args, **kwargs)
        _MilesRouter.forward = _miles_router_fwd
    except Exception as _e:
        logging.warning("expert_bias router-diag shim not applied: %s", _e)
    _miles_sys.stderr.write(">>> miles nemotron_h attn-shim: installed\n")
    _miles_sys.stderr.flush()
except Exception as _e:  # best-effort shim
    logging.warning("nemotron_h attn-unpack shim not applied: %s", _e)
