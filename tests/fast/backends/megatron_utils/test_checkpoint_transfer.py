import pickle

import pytest
import torch
from torch.utils._pytree import tree_flatten_with_path, tree_unflatten

from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict

from miles.backends.megatron_utils.checkpoint_transfer import (
    _deserialize_from_transport,
    _serialize_for_transport,
)


@pytest.fixture()
def state_dict() -> MCoreTensorAwareStateDict:
    sharded_state_dict: dict = {
        "model": {
            "layer1.weight": ShardedTensor.from_rank_offsets(
                "layer1.weight", torch.arange(32, dtype=torch.float32).reshape(4, 8)
            ),
            "layer2.weight": ShardedTensor.from_rank_offsets(
                "layer2.weight", torch.full((2, 6), fill_value=7.0)
            ),
        },
        "optimizer": {
            "step": ShardedTensor.from_rank_offsets("step", torch.tensor([100], dtype=torch.int64)),
        },
    }
    common = {"iteration": 0, "args_repr": "dummy"}
    return MCoreTensorAwareStateDict(common=common, sharded_state_dict=sharded_state_dict)


class TestSerializeForTransport:
    def test_returns_separated_tensors_iteration_and_hollow_shell(
        self, state_dict: MCoreTensorAwareStateDict
    ):
        original_tensors = [t.clone() for t in state_dict.tensors]

        payload = _serialize_for_transport(state_dict=state_dict, iteration=42)

        assert payload["iteration"] == 42
        assert isinstance(payload["tensors"], list)
        assert len(payload["tensors"]) == 3
        for t, original in zip(payload["tensors"], original_tensors, strict=True):
            assert torch.equal(t, original)
        assert payload["hollow_state_dict"] is state_dict
        assert payload["hollow_state_dict"].is_hollow

    def test_pytree_flatten_yields_each_tensor_as_separate_leaf(
        self, state_dict: MCoreTensorAwareStateDict
    ):
        """The whole point of the fix: PGTransport's tree_flatten_with_path must see
        each ShardedTensor.data as its own leaf, not buried inside a pickled blob."""
        payload = _serialize_for_transport(state_dict=state_dict, iteration=42)

        leaves, _ = tree_flatten_with_path(payload)
        tensor_leaves = [v for _, v in leaves if isinstance(v, torch.Tensor)]
        non_tensor_leaves = [v for _, v in leaves if not isinstance(v, torch.Tensor)]

        assert len(tensor_leaves) == 3
        assert any(isinstance(v, MCoreTensorAwareStateDict) for v in non_tensor_leaves)
        assert 42 in non_tensor_leaves

    def test_hollow_shell_pickles_without_dragging_tensor_data(
        self, state_dict: MCoreTensorAwareStateDict
    ):
        """PGTransport pickles non-tensor leaves; the hollow shell must survive
        a pickle round-trip and not contain any of the original tensor storage."""
        payload = _serialize_for_transport(state_dict=state_dict, iteration=42)

        restored = pickle.loads(pickle.dumps(payload["hollow_state_dict"]))

        assert restored.is_hollow
        sharded_tensors = list(restored._sharded_tensors)
        assert len(sharded_tensors) == 3
        assert all(sh.data is None for sh in sharded_tensors)
        assert all(hasattr(sh, "orig_device") for sh in sharded_tensors)


class TestDeserializeFromTransport:
    def test_round_trip_preserves_tensor_values_iteration_and_common(
        self, state_dict: MCoreTensorAwareStateDict
    ):
        original_tensors = [t.clone() for t in state_dict.tensors]
        original_common = dict(state_dict.common)

        payload = _serialize_for_transport(state_dict=state_dict, iteration=42)
        iteration_back, state_dict_back = _deserialize_from_transport(payload)

        assert iteration_back == 42
        assert not state_dict_back.is_hollow
        assert state_dict_back.common == original_common
        for original, back in zip(original_tensors, state_dict_back.tensors, strict=True):
            assert torch.equal(original, back)

    def test_full_pgtransport_simulation_round_trip(
        self, state_dict: MCoreTensorAwareStateDict
    ):
        """End-to-end simulation of PGTransport: pytree flatten on sender, pickle
        non-tensor leaves + treespec, clone tensor leaves to mimic NCCL transfer,
        unflatten on receiver. Verifies our (de)serializers survive the actual
        wire protocol — not just an in-process pop/insert."""
        original_tensors = [t.clone() for t in state_dict.tensors]
        original_common = dict(state_dict.common)

        # Step 1: sender — wrap state_dict for transport
        payload = _serialize_for_transport(state_dict=state_dict, iteration=42)

        # Step 2: sender — flatten via pytree (what PGTransport does internally)
        leaves, treespec = tree_flatten_with_path(payload)

        # Step 3: sender — pickle treespec + non-tensor leaves; "send" tensor leaves over the wire
        is_tensor_mask: list[bool] = [isinstance(v, torch.Tensor) for _, v in leaves]
        pickled_metadata = pickle.dumps(
            (treespec, [v for v, m in zip([v for _, v in leaves], is_tensor_mask) if not m])
        )
        wire_tensors = [v.clone() for v, m in zip([v for _, v in leaves], is_tensor_mask) if m]

        # Step 4: receiver — unpickle metadata + interleave received tensors back into leaf order
        treespec_recv, non_tensor_values = pickle.loads(pickled_metadata)
        recv_values: list = []
        ti = 0
        nti = 0
        for is_tensor in is_tensor_mask:
            if is_tensor:
                recv_values.append(wire_tensors[ti])
                ti += 1
            else:
                recv_values.append(non_tensor_values[nti])
                nti += 1
        payload_recv = tree_unflatten(recv_values, treespec_recv)

        # Step 5: receiver — unwrap into iteration + state_dict
        iteration_back, state_dict_back = _deserialize_from_transport(payload_recv)

        assert iteration_back == 42
        assert not state_dict_back.is_hollow
        assert state_dict_back.common == original_common
        for original, back in zip(original_tensors, state_dict_back.tensors, strict=True):
            assert torch.equal(original, back)
