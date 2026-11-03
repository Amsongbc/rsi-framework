"""Shared data contracts between blocks.

Decisions are what a policy hands the runner. LLM-only and Agent + VLA share them;
they differ in which decision kinds the assembled protocol accepts (an LLM-only
profile has no VLA, so a VLACall is rejected before anything moves).
"""
import dataclasses


@dataclasses.dataclass(frozen=True)
class Observe:
    note: str = ''


@dataclasses.dataclass(frozen=True)
class Finish:
    note: str = ''


@dataclasses.dataclass(frozen=True)
class PythonCell:
    """A program for the persistent namespace; tools are reached through `robo`."""
    code: str
    note: str = ''


@dataclasses.dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict = dataclasses.field(default_factory=dict)
    note: str = ''
    expected_change: str = ''


@dataclasses.dataclass(frozen=True)
class VLACall:
    """Hand `language` (None = task instruction) to the VLA; the protocol bounds the prefix."""
    language: str | None = None
    model_actions: int | None = None
    note: str = ''
    expected_change: str = ''


@dataclasses.dataclass(frozen=True)
class VisualAck:
    observation_index: int
    native_action_index: int
    image_sha256: dict
    viewed_by: str
    evidence: str
    note: str = ''


DECISIONS = (Observe, Finish, PythonCell, ToolCall, VLACall, VisualAck)


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    """A tool the policy may call. `kind` is readonly, motion or memory."""
    name: str
    kind: str
    description: str
    parameters: dict
    version: str = '1'

    def public(self):
        return dataclasses.asdict(self)


def decision_record(decision):
    """JSON form of a decision for records and data lineage."""
    return {'kind': type(decision).__name__, **dataclasses.asdict(decision)}


class ToolError(Exception):
    """Tool refusal for framework tool sets (same constructor shape as PyRUA's ToolError)."""

    def __init__(self, tool, message):
        super().__init__(f'{tool}: {message}')
        self.tool = tool
        self.message = message


class EpisodeEnded(Exception):
    """Raised by framework environment wrappers when the episode or its budget is over."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason
