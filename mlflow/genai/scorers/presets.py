import json
from dataclasses import asdict, dataclass, field
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer


_SERIALIZATION_VERSION = 1


@dataclass
class SerializedPreset:
    preset_name: str
    scorers: list[dict[str, Any]]
    version: int = 1
    mlflow_version: str = field(default_factory=lambda: mlflow.__version__)
    serialization_version: int = _SERIALIZATION_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "SerializedPreset":
        data = json.loads(json_str)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Preset:
    def __init__(self, name: str, scorers: list[Scorer]):
        self._name = name
        self._validate_no_duplicates(scorers)
        self._scorers = tuple(scorers)

    @staticmethod
    def _validate_no_duplicates(scorers: list[Scorer]):
        seen: set[tuple[type, str]] = set()
        for scorer in scorers:
            key = (type(scorer), scorer.name)
            if key in seen:
                raise MlflowException.invalid_parameter_value(
                    f"Duplicate scorer: {type(scorer).__name__} with name '{scorer.name}'. "
                    "Use different names for scorers of the same type."
                )
            seen.add(key)

    @property
    def name(self) -> str:
        return self._name

    @property
    def scorers(self) -> list[Scorer]:
        return list(self._scorers)

    def __iter__(self):
        return iter(self._scorers)

    def __len__(self):
        return len(self._scorers)

    def __repr__(self):
        scorer_names = [type(s).__name__ for s in self._scorers]
        return f"Preset('{self._name}', [{', '.join(scorer_names)}])"

    def model_dump(self) -> dict[str, Any]:
        serialized = SerializedPreset(
            preset_name=self._name,
            scorers=[scorer.model_dump() for scorer in self._scorers],
        )
        return asdict(serialized)

    @classmethod
    def model_validate(cls, obj: dict[str, Any] | str) -> "Preset":
        if isinstance(obj, str):
            data = json.loads(obj)
        else:
            data = obj

        serialized = SerializedPreset(
            **{k: v for k, v in data.items() if k in SerializedPreset.__dataclass_fields__}
        )

        scorers = [Scorer.model_validate(s) for s in serialized.scorers]
        return cls(name=serialized.preset_name, scorers=scorers)

    def register(self, *, experiment_id: str | None = None):
        from mlflow.genai.scorers.preset_registry import _get_preset_store

        store = _get_preset_store()
        serialized = json.dumps(self.model_dump())
        store.register_preset(experiment_id, self._name, serialized)

    def copy(self, *, to_experiment_id: str):
        from mlflow.genai.scorers.preset_registry import _get_preset_store

        store = _get_preset_store()
        serialized = json.dumps(self.model_dump())
        store.register_preset(to_experiment_id, self._name, serialized)


class Rag(Preset):
    def __init__(self):
        from mlflow.genai.scorers.builtin_scorers import (
            Completeness,
            RelevanceToQuery,
            RetrievalGroundedness,
            RetrievalRelevance,
            Safety,
        )

        super().__init__(
            "rag",
            [
                RetrievalRelevance(),
                RetrievalGroundedness(),
                RelevanceToQuery(),
                Safety(),
                Completeness(),
            ],
        )


class Agent(Preset):
    def __init__(self):
        from mlflow.genai.scorers.builtin_scorers import (
            Completeness,
            RelevanceToQuery,
            Safety,
            ToolCallCorrectness,
            ToolCallEfficiency,
        )

        super().__init__(
            "agent",
            [
                ToolCallCorrectness(),
                ToolCallEfficiency(),
                RelevanceToQuery(),
                Safety(),
                Completeness(),
            ],
        )


class ConversationalAgent(Preset):
    def __init__(self):
        from mlflow.genai.scorers.builtin_scorers import (
            Completeness,
            ConversationalSafety,
            ConversationalToolCallEfficiency,
            ConversationCompleteness,
            KnowledgeRetention,
            RelevanceToQuery,
            Safety,
            ToolCallCorrectness,
            ToolCallEfficiency,
            UserFrustration,
        )

        super().__init__(
            "conversational-agent",
            [
                ToolCallCorrectness(),
                ToolCallEfficiency(),
                RelevanceToQuery(),
                Safety(),
                Completeness(),
                UserFrustration(),
                ConversationCompleteness(),
                ConversationalSafety(),
                ConversationalToolCallEfficiency(),
                KnowledgeRetention(),
            ],
        )
