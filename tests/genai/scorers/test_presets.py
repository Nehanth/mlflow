import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import (
    Agent,
    ConversationalAgent,
    Preset,
    Rag,
)
from mlflow.genai.scorers.builtin_scorers import (
    BuiltInScorer,
    Completeness,
    ConversationalSafety,
    ConversationalToolCallEfficiency,
    ConversationCompleteness,
    Fluency,
    KnowledgeRetention,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    Safety,
    ToolCallCorrectness,
    ToolCallEfficiency,
    UserFrustration,
)
from mlflow.genai.scorers.presets import SerializedPreset
from mlflow.genai.scorers.validation import validate_scorers


class TestPresetClass:
    def test_preset_init(self):
        p = Preset("test", [Safety(), Fluency()])
        assert p.name == "test"
        assert len(p) == 2

    def test_preset_scorers_returns_list(self):
        p = Preset("test", [Safety()])
        assert isinstance(p.scorers, list)

    def test_preset_scorers_returns_fresh_copy(self):
        p = Preset("test", [Safety()])
        s1 = p.scorers
        s2 = p.scorers
        assert s1 is not s2

    def test_preset_iter(self):
        scorers = [Safety(), Fluency()]
        p = Preset("test", scorers)
        assert list(p) == scorers

    def test_preset_len(self):
        p = Preset("test", [Safety(), Fluency(), Completeness()])
        assert len(p) == 3

    def test_preset_repr(self):
        p = Preset("test", [Safety(), Fluency()])
        r = repr(p)
        assert "test" in r
        assert "Safety" in r
        assert "Fluency" in r

    def test_preset_blocks_duplicates(self):
        with pytest.raises(MlflowException, match="Duplicate scorer"):
            Preset("test", [Safety(), Safety()])

    def test_preset_allows_same_type_different_names(self):
        p = Preset("test", [Safety(name="s1"), Safety(name="s2")])
        assert len(p) == 2

    def test_preset_empty(self):
        p = Preset("empty", [])
        assert len(p) == 0
        assert p.scorers == []


class TestBuiltinPresets:
    @pytest.mark.parametrize(
        ("preset_cls", "expected_types"),
        [
            (
                Rag,
                {RetrievalRelevance, RetrievalGroundedness, RelevanceToQuery, Safety, Completeness},
            ),
            (
                Agent,
                {ToolCallCorrectness, ToolCallEfficiency, RelevanceToQuery, Safety, Completeness},
            ),
            (
                ConversationalAgent,
                {
                    ToolCallCorrectness,
                    ToolCallEfficiency,
                    RelevanceToQuery,
                    Safety,
                    Completeness,
                    UserFrustration,
                    ConversationCompleteness,
                    ConversationalSafety,
                    ConversationalToolCallEfficiency,
                    KnowledgeRetention,
                },
            ),
        ],
    )
    def test_builtin_preset_scorers(self, preset_cls, expected_types):
        preset = preset_cls()
        scorer_types = {type(s) for s in preset}
        assert scorer_types == expected_types

    @pytest.mark.parametrize("preset_cls", [Rag, Agent, ConversationalAgent])
    def test_builtin_preset_all_builtin_scorers(self, preset_cls):
        preset = preset_cls()
        assert all(isinstance(s, BuiltInScorer) for s in preset)

    @pytest.mark.parametrize("preset_cls", [Rag, Agent, ConversationalAgent])
    def test_builtin_preset_no_duplicates(self, preset_cls):
        preset = preset_cls()
        scorer_types = [type(s) for s in preset]
        assert len(scorer_types) == len(set(scorer_types))

    @pytest.mark.parametrize("preset_cls", [Rag, Agent, ConversationalAgent])
    def test_builtin_preset_fresh_instances(self, preset_cls):
        p1 = preset_cls()
        p2 = preset_cls()
        assert p1.scorers[0] is not p2.scorers[0]


class TestSerialization:
    def test_model_dump(self):
        p = Preset("test", [Safety(), Fluency()])
        dumped = p.model_dump()
        assert dumped["preset_name"] == "test"
        assert len(dumped["scorers"]) == 2
        assert "mlflow_version" in dumped
        assert "serialization_version" in dumped

    def test_model_validate_from_dict(self):
        p = Preset("test", [Safety(), Fluency()])
        dumped = p.model_dump()
        restored = Preset.model_validate(dumped)
        assert restored.name == "test"
        assert len(restored) == 2
        assert type(restored.scorers[0]).__name__ == "Safety"
        assert type(restored.scorers[1]).__name__ == "Fluency"

    def test_model_validate_from_json_string(self):
        import json

        p = Preset("test", [Safety()])
        json_str = json.dumps(p.model_dump())
        restored = Preset.model_validate(json_str)
        assert restored.name == "test"
        assert len(restored) == 1

    def test_round_trip_builtin_presets(self):
        for cls in [Rag, Agent]:
            original = cls()
            dumped = original.model_dump()
            restored = Preset.model_validate(dumped)
            assert restored.name == original.name
            assert len(restored) == len(original)

    def test_serialized_preset_to_json(self):
        sp = SerializedPreset(preset_name="test", scorers=[{"name": "safety"}])
        json_str = sp.to_json()
        restored = SerializedPreset.from_json(json_str)
        assert restored.preset_name == "test"
        assert len(restored.scorers) == 1


class TestValidateScorers:
    def test_flatten_preset_in_list(self):
        result = validate_scorers([Agent()])
        assert len(result) == 5
        assert all(isinstance(s, BuiltInScorer) for s in result)

    def test_flatten_preset_with_individual_scorer(self):
        result = validate_scorers([Agent(), Fluency()])
        assert len(result) == 6
        scorer_types = {type(s).__name__ for s in result}
        assert "Fluency" in scorer_types

    def test_dedup_across_preset_and_scorer(self):
        result = validate_scorers([Agent(), Safety()])
        scorer_names = [s.name for s in result]
        assert scorer_names.count("safety") == 1

    def test_dedup_across_multiple_presets(self):
        result = validate_scorers([Agent(), Rag()])
        scorer_names = [s.name for s in result]
        assert scorer_names.count("safety") == 1
        assert scorer_names.count("relevance_to_query") == 1
        assert scorer_names.count("completeness") == 1

    def test_preserves_different_names(self):
        result = validate_scorers([
            Preset("test", [Safety(name="s1"), Safety(name="s2")]),
        ])
        safety_scorers = [s for s in result if type(s).__name__ == "Safety"]
        assert len(safety_scorers) == 2
