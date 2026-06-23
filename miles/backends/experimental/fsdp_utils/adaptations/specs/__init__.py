"""Per-arch adaptation specs.

Importing this package registers every arch's hooks across the mechanism registries (weight_bridge,
class_patches, packing, post_load_fixups, precision). Adding a model = add a ``<arch>.py`` module here
that registers whatever hooks it needs, plus an import below.
"""

from . import glm4_moe_lite, nemotron_h, qwen3_5_moe, qwen3_moe  # noqa: F401  (imports trigger registration)
