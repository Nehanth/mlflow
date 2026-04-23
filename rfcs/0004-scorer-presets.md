---
start_date: 2026-04-23
mlflow_issue: https://github.com/mlflow/mlflow/issues/21445
rfc_pr:
---

# Scorer Presets for Common Evaluation Patterns

| Author(s)              | Nehanth                   |
| :--------------------- | :------------------------ |
| **Date Last Modified** | 2026-04-23                |
| **AI Assistant(s)**    | Claude Code               |

# Summary

> **Note:** This RFC is based on [mlflow/mlflow#21445](https://github.com/mlflow/mlflow/issues/21445). The motivation, proposed presets, and API examples in this document are derived from that issue, with additional design details, implementation specifics, and analysis added here.

MLflow provides 21 built-in scorers for evaluating GenAI outputs, but users have no way to select a coherent subset for a specific evaluation pattern. Today, evaluating an agent requires importing and instantiating 9+ individual scorer classes, and this boilerplate is copy-pasted across teams and templates.

This RFC proposes adding a `get_preset(name)` function to `mlflow.genai.scorers` that returns a curated list of built-in scorer instances for common evaluation patterns: `"rag"`, `"agent"`, `"conversational-agent"`, `"safety"`, and `"quality"`. A companion `list_presets()` function provides discoverability.

This is a thin, additive API layer (~60 lines) on top of existing scorer infrastructure. No new classes, no new abstractions, no breaking changes.

# Basic Example

**Simple usage:**

```python
import mlflow
from mlflow.genai.scorers import get_preset

result = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=get_preset("agent"),
)
```

**Extending a preset with custom scorers:**

```python
from mlflow.genai.scorers import get_preset, Guidelines

scorers = get_preset("agent") + [
    Guidelines(name="tone", guidelines=["Respond professionally"]),
]

result = mlflow.genai.evaluate(data=eval_dataset, scorers=scorers)
```

**Discovering available presets:**

```python
from mlflow.genai.scorers import list_presets

for name, scorer_names in list_presets().items():
    print(f"{name}: {', '.join(scorer_names)}")

# rag: RetrievalRelevance, RetrievalSufficiency, RetrievalGroundedness, ...
# agent: ToolCallCorrectness, ToolCallEfficiency, Safety, ...
# ...
```

## Motivation

### The Problem

As described in [the original issue](https://github.com/mlflow/mlflow/issues/21445), the Databricks agent app template [evaluate_agent.py](https://github.com/databricks/app-templates/blob/main/agent-openai-agents-sdk/agent_server/evaluate_agent.py) imports and instantiates 9 separate scorers to evaluate a conversational agent:

```python
from mlflow.genai.scorers import (
    Completeness,
    ConversationalSafety,
    ConversationCompleteness,
    Fluency,
    KnowledgeRetention,
    RelevanceToQuery,
    Safety,
    ToolCallCorrectness,
    UserFrustration,
)

mlflow.genai.evaluate(
    data=simulator,
    predict_fn=predict_fn,
    scorers=[
        Completeness(),
        ConversationCompleteness(),
        ConversationalSafety(),
        KnowledgeRetention(),
        UserFrustration(),
        Fluency(),
        RelevanceToQuery(),
        Safety(),
        ToolCallCorrectness(),
    ],
)
```

Every team building agent evaluation follows this pattern: import a list of scorers, instantiate them with empty constructors, and pass them as a list. This creates three problems (from the [original issue](https://github.com/mlflow/mlflow/issues/21445)):

1. **No built-in grouping.** `get_all_scorers()` returns all 19 default-constructible scorers. There is no way to get scorers filtered by evaluation pattern. Users evaluating a RAG pipeline receive `ToolCallCorrectness`; users evaluating an agent receive `RetrievalGroundedness`. Each unnecessary scorer wastes an LLM API call.

2. **21 scorers to choose from.** Users must read documentation for each scorer to determine relevance. Session-level scorers (e.g., `KnowledgeRetention`) only work with multi-turn evaluation, but nothing prevents users from passing them to single-turn evaluation where they silently produce no results.

3. **Copy-paste problem.** The same scorer lists get duplicated across templates, notebooks, and tutorials. When new scorers are added (like `ConversationalRoleAdherence`), existing copy-pasted lists don't pick them up.

### Who benefits

- **New users** get a curated starting point without reading all 21 scorer docs
- **Teams** get a canonical set of scorers for each evaluation pattern, eliminating drift across projects
- **Template authors** can use a single function call instead of maintaining scorer lists
- **MLflow maintainers** gain a single place to update when new scorers are added

### Out of Scope

- **Custom/user-defined presets.** Users can create their own lists by extending preset results. A persistent preset registry is a possible follow-up but not part of this proposal.
- **Parameterized presets.** Passing `model` or `inference_params` to all scorers in a preset (e.g., `get_preset("rag", model="openai:/gpt-4o")`) adds API complexity. Users can set these by iterating over the returned list.
- **Third-party scorer presets.** Integrating presets for DeepEval, RAGAS, or TruLens scorers is out of scope.
- **Preset registration/storage in the tracking server.** Presets are code-side only.

## Detailed Design

### Preset Definitions

The following table defines each preset, its contained scorers, and the rationale for inclusion/exclusion.

#### `"rag"` -- Retrieval-Augmented Generation

| Scorer | Why included |
|--------|-------------|
| RetrievalRelevance | Core RAG metric: are the retrieved chunks relevant to the query? |
| RetrievalSufficiency | Core RAG metric: do the retrieved chunks contain enough information? |
| RetrievalGroundedness | Core RAG metric: is the response grounded in retrieved context? |
| RelevanceToQuery | Response-level: does the final answer address the query? |
| Safety | Baseline safety check for any user-facing output. |
| Completeness | Does the response fully address the query given the context? |

**Excluded:** `Correctness` and `Equivalence` (require `expectations` data, which RAG evaluation often lacks). `ToolCallCorrectness/Efficiency` (not applicable to RAG). `Fluency` (secondary concern for factual retrieval tasks). `Summarization` (specialized to summarization tasks).

#### `"agent"` -- Single-Turn Tool-Calling Agent

| Scorer | Why included |
|--------|-------------|
| ToolCallCorrectness | Core agent metric: were the right tools called with the right arguments? |
| ToolCallEfficiency | Core agent metric: were there redundant or unnecessary tool calls? |
| RelevanceToQuery | Does the response address what the user asked? |
| Safety | Baseline safety check. |
| Completeness | Does the response fully resolve the user's request? |

**Excluded:** `RetrievalRelevance/Sufficiency/Groundedness` (not all agents do retrieval). `Fluency` (secondary for task-oriented agents). Session-level scorers (this preset is single-turn; use `"conversational-agent"` for multi-turn).

#### `"conversational-agent"` -- Multi-Turn Conversational Agent

| Scorer | Why included |
|--------|-------------|
| ToolCallCorrectness | Per-turn tool correctness. |
| ToolCallEfficiency | Per-turn tool efficiency. |
| RelevanceToQuery | Per-turn response relevance. |
| Safety | Per-turn safety. |
| Completeness | Per-turn completeness. |
| UserFrustration | Session-level: is the user frustrated? |
| ConversationCompleteness | Session-level: were all user requests addressed? |
| ConversationalSafety | Session-level: was safety maintained throughout? |
| ConversationalToolCallEfficiency | Session-level: tool efficiency across the conversation. |
| KnowledgeRetention | Session-level: does the agent remember prior context? |

This preset is a superset of `"agent"` plus all default-constructible session-level scorers.

**Excluded:** `ConversationalRoleAdherence` (requires a defined role/persona, not always present). `ConversationalGuidelines` (requires constructor arg `guidelines`). `Fluency` (secondary for conversational agents).

#### `"safety"` -- Safety-Focused Evaluation

| Scorer | Why included |
|--------|-------------|
| Safety | Single-turn safety evaluation. |
| ConversationalSafety | Session-level safety evaluation. |

Intentionally minimal and composable. Use alongside other presets: `get_preset("rag") + get_preset("safety")` deduplicates at the user's discretion.

#### `"quality"` -- General Output Quality

| Scorer | Why included |
|--------|-------------|
| RelevanceToQuery | Is the response on-topic? |
| Fluency | Is the response well-written? |
| Completeness | Is the response thorough? |
| Correctness | Is the response factually correct? (Requires `expectations`.) |

Architecture-independent scorers that evaluate the output text itself.

**Excluded:** `Equivalence` (too specialized; requires ground truth for semantic comparison). `Summarization` (too specialized for a general quality preset).

### API

```python
from typing import Literal


def get_preset(
    name: Literal["rag", "agent", "conversational-agent", "safety", "quality"],
) -> list[BuiltInScorer]:
    """Return a curated list of scorer instances for a common evaluation pattern.

    Each call returns fresh scorer instances. The returned list is mutable and
    can be extended with additional scorers using standard list operations.

    Args:
        name: The evaluation pattern to get scorers for.

    Returns:
        A list of BuiltInScorer instances.

    Raises:
        MlflowException: If name is not a valid preset.
    """


def list_presets() -> dict[str, list[str]]:
    """Return a mapping of preset names to their scorer class names.

    Returns:
        A dictionary where keys are preset names and values are lists
        of scorer class names (strings) contained in each preset.
    """
```

**Design choices:**

1. **`Literal` type for `name`** -- follows the project's Python style guide (`.claude/rules/python.md`) which mandates `Literal` for fixed-string parameters. Enables IDE autocompletion and type checking.

2. **Returns `list[BuiltInScorer]`** -- a plain mutable list, consistent with `get_all_scorers()`. Users can extend with `+`, filter with list comprehension, or modify in-place.

3. **Fresh instances per call** -- each invocation creates new scorer instances with default parameters. No shared mutable state. Matches `get_all_scorers()` behavior.

4. **`list_presets()` returns `dict[str, list[str]]`** -- lets users inspect presets without instantiating scorers or importing classes. Returns class names as strings for human-readable output.

5. **`MlflowException` for invalid names** -- follows the pattern in `validation.py`. Error message includes all valid preset names.

### Implementation

#### New file: `mlflow/genai/scorers/presets.py`

```python
from typing import Literal

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.builtin_scorers import (
    Completeness,
    ConversationalSafety,
    ConversationalToolCallEfficiency,
    ConversationCompleteness,
    Correctness,
    Fluency,
    KnowledgeRetention,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
    Safety,
    ToolCallCorrectness,
    ToolCallEfficiency,
    UserFrustration,
)

_PRESETS: dict[str, list[type]] = {
    "rag": [
        RetrievalRelevance,
        RetrievalSufficiency,
        RetrievalGroundedness,
        RelevanceToQuery,
        Safety,
        Completeness,
    ],
    "agent": [
        ToolCallCorrectness,
        ToolCallEfficiency,
        RelevanceToQuery,
        Safety,
        Completeness,
    ],
    "conversational-agent": [
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
    ],
    "safety": [
        Safety,
        ConversationalSafety,
    ],
    "quality": [
        RelevanceToQuery,
        Fluency,
        Completeness,
        Correctness,
    ],
}

_VALID_PRESET_NAMES = ", ".join(sorted(_PRESETS.keys()))
PresetName = Literal["rag", "agent", "conversational-agent", "safety", "quality"]


def get_preset(name: PresetName) -> list:
    if name not in _PRESETS:
        raise MlflowException.invalid_parameter_value(
            f"Unknown preset '{name}'. Valid presets are: {_VALID_PRESET_NAMES}"
        )
    return [scorer_class() for scorer_class in _PRESETS[name]]


def list_presets() -> dict[str, list[str]]:
    return {
        name: [cls.__name__ for cls in classes]
        for name, classes in _PRESETS.items()
    }
```

No circular dependency risk: `presets.py` imports from `builtin_scorers.py`, and nothing in the existing chain imports from `presets.py`.

#### Updated: `mlflow/genai/scorers/__init__.py`

Add `"get_preset"` and `"list_presets"` to `_LAZY_IMPORTS`, `__all__`, and the `TYPE_CHECKING` block. The `__getattr__` function needs to dispatch to the `presets` module:

```python
_LAZY_IMPORTS_PRESETS = {"get_preset", "list_presets"}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        from mlflow.genai.scorers import builtin_scorers
        return getattr(builtin_scorers, name)
    if name in _LAZY_IMPORTS_PRESETS:
        from mlflow.genai.scorers import presets
        return getattr(presets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

#### Updated: `mlflow/genai/__init__.py`

Re-export for convenience:

```python
from mlflow.genai.scorers import get_preset, list_presets
```

#### Updated: `mlflow/genai/scorers/validation.py`

Update the error message in `validate_scorers()` to suggest presets:

```python
raise MlflowException.invalid_parameter_value(
    "The `scorers` argument must be a list of scorers. If you are unsure about which "
    "scorer to use, you can specify `scorers=get_preset('agent')` for a curated set, "
    "or `scorers=get_all_scorers()` for all available built-in scorers."
)
```

### Testing Plan

New test file: `tests/genai/scorers/test_presets.py`

| Test | What it verifies |
|------|-----------------|
| `test_get_preset_rag` | Returns exactly the 6 RAG scorers by class type |
| `test_get_preset_agent` | Returns exactly the 5 agent scorers by class type |
| `test_get_preset_conversational_agent` | Returns exactly the 10 conversational-agent scorers |
| `test_get_preset_safety` | Returns exactly the 2 safety scorers |
| `test_get_preset_quality` | Returns exactly the 4 quality scorers |
| `test_get_preset_returns_fresh_instances` | Two calls return different objects (`is not`) |
| `test_get_preset_returns_mutable_list` | Returned list can be extended with `+` |
| `test_get_preset_invalid_name` | Raises `MlflowException` with valid names in message |
| `test_list_presets` | Returns correct dict structure with correct class names |
| `test_preset_scorers_are_valid` | Every scorer in every preset passes `validate_scorers()` |
| `test_preset_no_overlap_rag_agent` | RAG and agent presets have no unexpected overlap |

Tests follow the parametrize pattern from `.claude/rules/python.md`:

```python
@pytest.mark.parametrize("preset_name", ["rag", "agent", "conversational-agent", "safety", "quality"])
def test_get_preset_returns_scorer_instances(preset_name):
    scorers = get_preset(preset_name)
    assert all(isinstance(s, BuiltInScorer) for s in scorers)
    assert len(scorers) == len(set(type(s) for s in scorers))  # no duplicates
```

### Files Changed

| File | Change |
|------|--------|
| `mlflow/genai/scorers/presets.py` | **New.** Preset definitions, `get_preset()`, `list_presets()`. |
| `mlflow/genai/scorers/__init__.py` | Add lazy imports for `get_preset`, `list_presets`. |
| `mlflow/genai/__init__.py` | Re-export `get_preset`, `list_presets`. |
| `mlflow/genai/scorers/validation.py` | Update error message to suggest presets. |
| `tests/genai/scorers/test_presets.py` | **New.** Tests for all presets. |

## Drawbacks

1. **Opinionated defaults.** Not everyone will agree on which scorers belong in which preset. Users may disagree that Safety belongs in `"rag"` or that Fluency does not belong in `"agent"`. Mitigation: presets return mutable lists, so users can freely add or remove scorers.

2. **Maintenance burden.** When new scorers are added to MLflow, maintainers must decide which presets (if any) to add them to. However, the preset definitions are ~30 lines of declarative code -- the cost is low.

3. **Implicit behavior changes on upgrade.** If MLflow 3.13 adds a scorer to the `"agent"` preset, users upgrading from 3.12 will silently get different evaluation results (an additional score column). Mitigation: this is standard behavior for `get_all_scorers()` already. Document clearly that preset contents may evolve across releases.

4. **Potential for stale presets.** If the preset definitions fall out of sync with best practices, they could mislead users. Mitigation: presets are defined alongside the scorers in the same package, reviewed in the same PRs.

# Alternatives

### 1. Tag-based filtering

Add category tags to each scorer class (e.g., `categories = {"rag", "safety"}`) and provide `get_scorers(categories=["rag"])`.

**Pros:** More flexible; users can query by arbitrary combinations. Each scorer self-declares its categories.

**Cons:** Over-engineered for 21 scorers. Requires modifying every existing scorer class. Categories are not always orthogonal (is `Safety` in "rag" or "safety" or both?).

**Decision:** Presets are simpler and solve the stated problem without modifying existing scorer classes.

### 2. Enum-based API

```python
class ScorerPreset(Enum):
    RAG = "rag"
    AGENT = "agent"
    ...

scorers = ScorerPreset.RAG.get_scorers()
```

**Pros:** Type-safe. Discoverable via IDE.

**Cons:** Heavier API surface. Enum methods feel unusual. The `Literal` type on `get_preset` already provides IDE autocompletion.

**Decision:** A simple function with `Literal` type achieves the same discoverability with less API surface.

### 3. EvaluationSuite class

Bundle scorers + dataset + config into a single `EvaluationSuite` object:

```python
suite = EvaluationSuite("agent", data=dataset, predict_fn=fn)
result = suite.run()
```

**Pros:** More powerful; could encapsulate the full evaluation workflow.

**Cons:** Much higher complexity. Overlaps with `mlflow.genai.evaluate()`. Could be a follow-up if presets prove useful.

**Decision:** Too heavy for the stated problem. Presets are a better starting point.

### 4. Do nothing

Users continue copy-pasting scorer lists from templates and documentation.

**Pros:** No code to maintain.

**Cons:** Does not scale as the scorer count grows past 21. New users have no guidance. Copy-pasted lists become stale.

# Adoption Strategy

This is an **additive, non-breaking change**. No existing API is modified.

- **Existing users** can adopt presets at their own pace. Their current code continues to work unchanged.
- **Documentation and templates** should be updated to show `get_preset()` alongside the manual import pattern.
- **The `validate_scorers()` error message** should be updated to mention presets, helping users discover the feature when they make common mistakes.
- **Databricks agent templates** can simplify their evaluation code from 9 imports + 9 instantiations to a single `get_preset("conversational-agent")` call.

# Open Questions

1. **Should `get_preset` accept multiple names?** For example, `get_preset("rag", "safety")` returning a merged, deduplicated list. The simple alternative is `get_preset("rag") + get_preset("safety")`, which is explicit and requires no deduplication logic. **Recommendation:** Start with single name. Composition via `+` is sufficient and more transparent.

2. **Should `ConversationalRoleAdherence` be in `"conversational-agent"`?** It evaluates whether the agent maintains its assigned role, which requires a defined persona in the system prompt. Not all agents have this. The current proposal excludes it to avoid silent null results. **Open for discussion.**

3. **Should `Correctness` be in `"agent"` or `"rag"`?** It requires `expectations` data. Including it means the preset produces null results when expectations are missing, but the existing validation system already handles this gracefully (logs an info message). The current proposal includes it only in `"quality"` to keep other presets expectations-free. **Open for discussion.**

4. **Should there be an `"all"` preset?** `get_all_scorers()` already fills this role. Adding `"all"` creates two ways to do the same thing with no added value. **Recommendation:** Do not add.

5. **Future: parameterized presets?** A follow-up could add `get_preset("rag", model="openai:/gpt-4o")` to set the judge model for all returned scorers. This is deferred to keep the initial API simple.
