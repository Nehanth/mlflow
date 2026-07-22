import json
from functools import cached_property

from mlflow.entities._mlflow_object import _MlflowObject


class PresetVersion(_MlflowObject):
    def __init__(
        self,
        experiment_id: str,
        preset_name: str,
        preset_version: int,
        serialized_preset: str,
        creation_time: int,
        preset_id: str | None = None,
    ):
        self._experiment_id = experiment_id
        self._preset_name = preset_name
        self._preset_version = preset_version
        self._serialized_preset = serialized_preset
        self._creation_time = creation_time
        self._preset_id = preset_id

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def preset_name(self):
        return self._preset_name

    @property
    def preset_version(self):
        return self._preset_version

    @cached_property
    def serialized_preset(self):
        # Lazy deserialization: only parse JSON when first accessed
        return json.loads(self._serialized_preset)

    @property
    def creation_time(self):
        return self._creation_time

    @property
    def preset_id(self):
        return self._preset_id

    @classmethod
    def from_proto(cls, proto):
        # TODO: Implement once proto is regenerated (Preset message in service.proto)
        raise NotImplementedError("Proto for Preset has not been regenerated yet")

    def to_proto(self):
        # TODO: Implement once proto is regenerated (Preset message in service.proto)
        raise NotImplementedError("Proto for Preset has not been regenerated yet")

    def __repr__(self):
        return (
            f"<PresetVersion(experiment_id={self.experiment_id}, "
            f"preset_name='{self.preset_name}', "
            f"preset_version={self.preset_version})>"
        )
