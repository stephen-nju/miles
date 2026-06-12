import argparse
import json

import pytest
from pydantic import ValidationError

from miles.utils.test_utils.engine_kill_schedule import EngineKillAction, EngineKillScheduleExecutor


def _make_args(schedule: str | None) -> argparse.Namespace:
    return argparse.Namespace(ci_engine_kill_schedule=schedule)


def test_from_args_without_schedule_is_disabled():
    """Executor built from unset arg reports disabled and matches nothing."""
    executor = EngineKillScheduleExecutor.from_args(_make_args(None))

    assert not executor.enabled
    assert executor.actions_at(2) == []


def test_from_args_parses_schedule_and_matches_by_rollout():
    """Executor parses a JSON schedule and returns only the entries of the queried rollout."""
    schedule = json.dumps(
        [
            {"at_rollout": 3, "engine_index": 1},
            {"at_rollout": 7},
            {"at_rollout": 7, "engine_index": 2},
        ]
    )
    executor = EngineKillScheduleExecutor.from_args(_make_args(schedule))

    assert executor.enabled
    assert executor.actions_at(3) == [EngineKillAction(at_rollout=3, engine_index=1)]
    assert executor.actions_at(7) == [
        EngineKillAction(at_rollout=7, engine_index=0),
        EngineKillAction(at_rollout=7, engine_index=2),
    ]
    assert executor.actions_at(5) == []


def test_from_args_rejects_unknown_fields():
    """Schedule entries with unknown fields fail strict validation instead of being ignored."""
    schedule = json.dumps([{"at_rollout": 3, "engine": 1}])

    with pytest.raises(ValidationError):
        EngineKillScheduleExecutor.from_args(_make_args(schedule))
