"""glm4_moe_lite (GLM-4.7-Flash) adaptations.

Two hooks: the train->rollout weight transform and the fp32 master. Glm4MoeLiteNaiveMoe stores
experts.gate_up_proj [E,2I,H] (rows [gate|up]) + experts.down_proj [E,H,I] -- the IDENTICAL batched
layout as qwen3_moe -- so it reuses the same split transform. And its FSDP2 bf16 reshard perturbs
weights ~1 ULP vs a clean disk load, so it keeps an fp32 master (downcast back at weight sync).
"""

from ..precision import register_fp32_master_type
from ..weight_bridge import _qwen3_moe_expand, _qwen3_moe_matches, register_param_transform

register_param_transform("glm4_moe_lite", _qwen3_moe_matches, _qwen3_moe_expand)
register_fp32_master_type("glm4_moe_lite")
