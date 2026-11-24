"""Experiment contract models used by the Marimo server utilities.

These models are intentionally experiment-agnostic.  The package remains under
``utils.marimo`` because the contract is consumed by the Marimo server
launcher, but it does not assume YOLO, a particular dataset, or one source
repository.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SourceSpec:
    """Immutable source provenance for a project or external repository."""

    kind: str
    repo: str
    commit: str
    remote: str | None = None
    dirty: str = ""
    patch_sha256: str | None = None
    environment_sha256: str | None = None

    @classmethod
    def from_git(
        cls,
        *,
        repo: Path,
        commit: str,
        dirty: str = "",
        remote: str | None = None,
        patch_sha256: str | None = None,
        environment_sha256: str | None = None,
    ) -> "SourceSpec":
        return cls(
            kind="git",
            repo=str(repo.resolve()),
            commit=commit,
            remote=remote,
            dirty=dirty,
            patch_sha256=patch_sha256,
            environment_sha256=environment_sha256,
        )


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset provenance shared by baseline and reproduction runs."""

    name: str
    root: str
    yaml: str
    split_seed: int | None = None
    protocol: str | None = None


@dataclass(frozen=True)
class ExecutionSpec:
    """Exact execution identity without assuming a model framework."""

    command: tuple[str, ...]
    cwd: str
    seed: int | None = None
    workers: int | None = None
    epochs: int | None = None
    patience: int | None = None


@dataclass(frozen=True)
class OutputSpec:
    """Local and remote output destinations for one Marimo run."""

    artifact_root: str
    hf_repo_id: str | None = None
    remote_prefix: str | None = None


@dataclass(frozen=True)
class ExperimentContract:
    """Generic experiment identity carried by a Marimo launch.

    Framework-specific values belong in ``extra``.  This keeps the core
    contract usable for baselines, project experiments, and external Git
    reproductions without weakening the Marimo lifecycle checks.
    """

    run_id: str
    experiment_kind: str
    source: SourceSpec
    dataset: DatasetSpec
    execution: ExecutionSpec
    outputs: OutputSpec
    backend: str = "marimo"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend != "marimo":
            raise ValueError(f"Marimo utilities require backend='marimo', got {self.backend!r}")
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.experiment_kind.strip():
            raise ValueError("experiment_kind must not be empty")
        if not self.execution.command:
            raise ValueError("execution.command must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible contract data with resolved path strings."""
        value = asdict(self)
        value["execution"]["command"] = list(self.execution.command)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentContract":
        """Load the generic nested contract emitted by ``to_dict``."""
        source = SourceSpec(**dict(value["source"]))
        dataset = DatasetSpec(**dict(value["dataset"]))
        execution_data = dict(value["execution"])
        execution_data["command"] = tuple(execution_data["command"])
        execution = ExecutionSpec(**execution_data)
        outputs = OutputSpec(**dict(value["outputs"]))
        return cls(
            run_id=str(value["run_id"]),
            experiment_kind=str(value["experiment_kind"]),
            source=source,
            dataset=dataset,
            execution=execution,
            outputs=outputs,
            backend=str(value.get("backend", "marimo")),
            extra=dict(value.get("extra", {})),
        )
