import argparse
import logging

from pydantic import TypeAdapter

from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


class EngineKillAction(FrozenStrictBaseModel):
    at_rollout: int
    engine_index: int = 0


_ACTION_LIST_ADAPTER: TypeAdapter[list[EngineKillAction]] = TypeAdapter(list[EngineKillAction])


class EngineKillScheduleExecutor:
    """Schedule-driven engine crash injection for FT tests (--ci-engine-kill-schedule)."""

    def __init__(self, *, actions: list[EngineKillAction]) -> None:
        self._actions = actions

    @staticmethod
    def from_args(args: argparse.Namespace) -> "EngineKillScheduleExecutor":
        raw: str | None = args.ci_engine_kill_schedule
        if not raw:
            return EngineKillScheduleExecutor(actions=[])

        actions = _ACTION_LIST_ADAPTER.validate_json(raw)
        logger.info("Engine kill schedule activated: %d kills", len(actions))
        return EngineKillScheduleExecutor(actions=actions)

    @property
    def enabled(self) -> bool:
        return bool(self._actions)

    def actions_at(self, rollout_id: int) -> list[EngineKillAction]:
        return [action for action in self._actions if action.at_rollout == rollout_id]
