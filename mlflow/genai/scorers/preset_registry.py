import json
from abc import ABCMeta, abstractmethod

from mlflow.genai.scorers.presets import Preset
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.uri import get_uri_scheme


class AbstractPresetStore(metaclass=ABCMeta):
    @abstractmethod
    def register_preset(self, experiment_id: str | None, name: str, serialized_preset: str):
        pass

    @abstractmethod
    def list_presets(self, experiment_id: str | None) -> list["Preset"]:
        pass

    @abstractmethod
    def get_preset(self, experiment_id: str | None, name: str, version: int | None = None) -> "Preset":
        pass

    @abstractmethod
    def delete_preset(self, experiment_id: str | None, name: str, version: int | str | None = None):
        pass


class MlflowTrackingPresetStore(AbstractPresetStore):
    def __init__(self, tracking_uri: str):
        self._tracking_store = _get_store(tracking_uri)

    def register_preset(self, experiment_id: str | None, name: str, serialized_preset: str):
        experiment_id = experiment_id or _get_experiment_id()
        return self._tracking_store.register_preset(experiment_id, name, serialized_preset)

    def list_presets(self, experiment_id: str | None) -> list[Preset]:
        experiment_id = experiment_id or _get_experiment_id()
        preset_versions = self._tracking_store.list_presets(experiment_id)
        return [
            Preset.model_validate(pv.serialized_preset)
            for pv in preset_versions
        ]

    def get_preset(self, experiment_id: str | None, name: str, version: int | None = None) -> Preset:
        experiment_id = experiment_id or _get_experiment_id()
        preset_version = self._tracking_store.get_preset(experiment_id, name, version)
        return Preset.model_validate(preset_version.serialized_preset)

    def delete_preset(self, experiment_id: str | None, name: str, version: int | str | None = None):
        experiment_id = experiment_id or _get_experiment_id()
        self._tracking_store.delete_preset(experiment_id, name, version)


def _get_preset_store(tracking_uri: str | None = None) -> AbstractPresetStore:
    from mlflow.tracking._tracking_service.utils import _get_tracking_uri

    tracking_uri = tracking_uri or _get_tracking_uri()
    return MlflowTrackingPresetStore(tracking_uri)


def get_scorer_preset(
    *, name: str, experiment_id: str | None = None, version: int | None = None
) -> Preset:
    store = _get_preset_store()
    return store.get_preset(experiment_id, name, version)


def list_scorer_presets(*, experiment_id: str | None = None) -> list[Preset]:
    store = _get_preset_store()
    return store.list_presets(experiment_id)


def delete_scorer_preset(
    *, name: str, experiment_id: str | None = None, version: int | str | None = None
) -> None:
    store = _get_preset_store()
    store.delete_preset(experiment_id, name, version)
