from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dub_server.artifact_cleanup import (
    ArtifactCleanupError,
    execute_cleanup,
    plan_artifact_cleanup,
)
from dub_server.state import JobStatus


@dataclass(frozen=True)
class FakeJob:
    id: str
    status: JobStatus
    retryable: bool
    updated_at: str
    revision: int = 0


class FakeStateStore:
    def __init__(self, jobs: dict[str, FakeJob]) -> None:
        self.jobs = jobs

    def run_if_job_revision(self, job_id, expected_revision, operation):
        current = self.jobs[job_id]
        if current.revision != expected_revision:
            return False
        return bool(operation(current))


def _job(
    identifier: str,
    status: JobStatus,
    *,
    retryable: bool = False,
    age_days: int = 0,
    now: datetime,
) -> FakeJob:
    return FakeJob(
        id=identifier,
        status=status,
        retryable=retryable,
        updated_at=(now - timedelta(days=age_days)).isoformat(),
    )


def test_cleanup_is_dry_run_by_default_and_respects_retention(tmp_path: Path) -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    jobs_root = tmp_path / "jobs"
    output_root = tmp_path / "output"
    jobs_root.mkdir()
    output_root.mkdir()
    cancelled = "cancelled-job"
    expired = "expired-retry"
    fresh = "fresh-retry"
    complete = "completed-job"
    for identifier in (cancelled, expired, fresh, complete):
        directory = jobs_root / identifier
        directory.mkdir()
        (directory / "artifact.bin").write_bytes(b"1234")
        (output_root / f"{identifier}.mp4").write_bytes(b"video")
    (output_root / f".{cancelled}.part.mp4").write_bytes(b"partial")
    (output_root / f".{cancelled}.vi.srt.random.part").write_text(
        "partial", encoding="utf-8"
    )

    plan = plan_artifact_cleanup(
        [
            _job(cancelled, JobStatus.CANCELLED, now=now),
            _job(expired, JobStatus.FAILED, retryable=True, age_days=8, now=now),
            _job(fresh, JobStatus.FAILED, retryable=True, age_days=2, now=now),
            _job(complete, JobStatus.COMPLETED, now=now),
        ],
        jobs_root=jobs_root,
        output_root=output_root,
        now=now,
    )
    selected = {action.job_id for action in plan.actions}
    assert selected == {cancelled, expired}
    assert plan.total_size_bytes > 0

    result = execute_cleanup(plan)
    assert result.dry_run is True
    assert result.removed == ()
    assert (jobs_root / cancelled / "artifact.bin").exists()
    assert (output_root / f"{expired}.mp4").exists()


def test_apply_removes_only_planned_paths_and_never_incoming(tmp_path: Path) -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    jobs_root = tmp_path / "jobs"
    output_root = tmp_path / "output"
    incoming_root = tmp_path / "incoming"
    for root in (jobs_root, output_root, incoming_root):
        root.mkdir()
    identifier = "cancelled-job"
    (jobs_root / identifier).mkdir()
    (jobs_root / identifier / "checkpoint.json").write_text("{}", encoding="utf-8")
    (output_root / f"{identifier}.mp4").write_bytes(b"incomplete")
    incoming = incoming_root / f"{identifier}.mkv"
    incoming.write_bytes(b"source-must-survive")
    unrelated = output_root / "another-job.mp4"
    unrelated.write_bytes(b"keep")

    plan = plan_artifact_cleanup(
        [job := _job(identifier, JobStatus.CANCELLED, now=now)],
        jobs_root=jobs_root,
        output_root=output_root,
        now=now,
    )
    result = execute_cleanup(
        plan,
        apply=True,
        state_store=FakeStateStore({identifier: job}),  # type: ignore[arg-type]
    )

    assert result.errors == ()
    assert len(result.removed) == 2
    assert not (jobs_root / identifier).exists()
    assert not (output_root / f"{identifier}.mp4").exists()
    assert incoming.read_bytes() == b"source-must-survive"
    assert unrelated.read_bytes() == b"keep"


def test_unsafe_job_id_and_filesystem_root_are_refused(tmp_path: Path) -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    jobs_root = tmp_path / "jobs"
    output_root = tmp_path / "output"
    jobs_root.mkdir()
    output_root.mkdir()
    plan = plan_artifact_cleanup(
        [_job("../escape", JobStatus.CANCELLED, now=now)],
        jobs_root=jobs_root,
        output_root=output_root,
        now=now,
    )
    assert plan.actions == ()
    assert "ID không an toàn" in plan.skipped[0]

    with pytest.raises(ArtifactCleanupError, match="filesystem root"):
        plan_artifact_cleanup(
            [],
            jobs_root=Path(Path.cwd().anchor),
            output_root=output_root,
            now=now,
        )


def test_symlink_job_directory_is_unlinked_without_touching_target(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink is unavailable")
    now = datetime(2026, 8, 1, tzinfo=UTC)
    jobs_root = tmp_path / "jobs"
    output_root = tmp_path / "output"
    outside = tmp_path / "outside"
    jobs_root.mkdir()
    output_root.mkdir()
    outside.mkdir()
    protected = outside / "protected.bin"
    protected.write_bytes(b"keep")
    link = jobs_root / "cancelled-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not permitted")

    plan = plan_artifact_cleanup(
        [job := _job("cancelled-link", JobStatus.CANCELLED, now=now)],
        jobs_root=jobs_root,
        output_root=output_root,
        now=now,
    )
    result = execute_cleanup(
        plan,
        apply=True,
        state_store=FakeStateStore({"cancelled-link": job}),  # type: ignore[arg-type]
    )

    assert result.errors == ()
    assert not link.exists()
    assert protected.read_bytes() == b"keep"


def test_apply_revalidates_revision_and_status_before_deleting(tmp_path: Path) -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    jobs_root = tmp_path / "jobs"
    output_root = tmp_path / "output"
    jobs_root.mkdir()
    output_root.mkdir()
    identifier = "retry-race"
    artifact = jobs_root / identifier
    artifact.mkdir()
    (artifact / "checkpoint.json").write_text("{}", encoding="utf-8")
    planned_job = _job(
        identifier,
        JobStatus.FAILED,
        retryable=True,
        age_days=8,
        now=now,
    )
    plan = plan_artifact_cleanup(
        [planned_job],
        jobs_root=jobs_root,
        output_root=output_root,
        now=now,
    )
    resumed_job = FakeJob(
        id=identifier,
        status=JobStatus.SYNTHESIZING,
        retryable=False,
        updated_at=now.isoformat(),
        revision=planned_job.revision + 1,
    )

    result = execute_cleanup(
        plan,
        apply=True,
        state_store=FakeStateStore({identifier: resumed_job}),  # type: ignore[arg-type]
    )

    assert result.removed == ()
    assert result.errors and "trạng thái đã thay đổi" in result.errors[0]
    assert (artifact / "checkpoint.json").is_file()
