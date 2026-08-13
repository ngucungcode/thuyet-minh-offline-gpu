from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from dub_server.domain import TranscriptSegment, TranscriptionResult
from dub_server.state import (
    ActiveJobExists,
    InvalidTransition,
    JobNotFound,
    JobStage,
    JobStatus,
    StateError,
    StateStore,
)


def _transcription_result(*, text: str = "Hello world") -> TranscriptionResult:
    return TranscriptionResult(
        source="asr",
        language="en",
        language_probability=0.91,
        duration_us=3_000_000,
        model_id="asr-faster-whisper-small",
        segments=(
            TranscriptSegment(
                start_us=100_000,
                end_us=1_250_000,
                text=text,
                average_log_probability=-0.3,
                no_speech_probability=0.02,
            ),
            TranscriptSegment(
                start_us=1_250_000,
                end_us=2_900_000,
                text="Second line",
            ),
        ),
    )


def test_state_store_uses_wal_and_persists_job_events(tmp_path: Path) -> None:
    database = tmp_path / "state" / "jobs.sqlite3"
    store = StateStore(database)

    assert store.journal_mode() == "wal"
    created = store.create_job(
        "release-1",
        {"rights_confirmed": True, "source_language": "auto"},
        job_id="job-1",
    )
    downloading = store.update_status(
        created.id,
        JobStatus.DOWNLOADING,
        details={"task_id": "abc", "downloaded_bytes": 10},
        progress_permille=125,
    )

    reopened = StateStore(database)
    persisted = reopened.get_job("job-1")
    assert persisted.status == JobStatus.DOWNLOADING
    assert persisted.progress_permille == 125
    assert persisted.details["task_id"] == "abc"
    events = reopened.list_events("job-1")
    assert [event.event_type for event in events] == ["job.created", "job.status"]
    assert events[-1].payload["progress_permille"] == 125
    assert downloading.revision == 1


def test_checkpoint_is_upserted_without_losing_other_job_state(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job("release-1", {"rights_confirmed": True})

    store.save_checkpoint(job.id, JobStage.ACQUISITION, {"bytes": 100})
    store.save_checkpoint(job.id, JobStage.ACQUISITION, {"bytes": 250})

    checkpoint = store.get_checkpoint(job.id, JobStage.ACQUISITION)
    assert checkpoint is not None
    assert checkpoint.payload == {"bytes": 250}
    assert len(store.list_events(job.id)) == 3


def test_create_ready_offline_job_commits_handoff_atomically(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")

    job = store.create_ready_offline_job(
        "local-upload:job-local",
        {"source_kind": "local_upload", "timing_profile": "natural"},
        {"source_media_path": "/data/incoming/job-local/source.mkv"},
        acquisition_checkpoint={"source": "local_upload"},
        subtitle_checkpoint={"transcript_source": "asr"},
        job_id="job-local",
    )

    assert job.status is JobStatus.READY_OFFLINE
    assert job.stage is JobStage.SUBTITLE
    assert job.progress_permille == 250
    assert store.get_checkpoint(job.id, JobStage.ACQUISITION).payload == {
        "source": "local_upload"
    }
    assert store.get_checkpoint(job.id, JobStage.SUBTITLE).payload == {
        "transcript_source": "asr"
    }
    assert [event.event_type for event in store.list_events(job.id)] == [
        "job.created",
        "job.checkpoint",
        "job.checkpoint",
    ]


def test_create_ready_offline_job_rolls_back_every_row_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")

    def fail_event(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("simulated event failure")

    monkeypatch.setattr(store, "_insert_event", fail_event)
    with pytest.raises(RuntimeError, match="simulated event failure"):
        store.create_ready_offline_job(
            "local-upload:job-local",
            {"source_kind": "local_upload"},
            {"source_media_path": "/data/incoming/job-local/source.mkv"},
            acquisition_checkpoint={"source": "local_upload"},
            subtitle_checkpoint={"transcript_source": "asr"},
            job_id="job-local",
        )

    with pytest.raises(JobNotFound):
        store.get_job("job-local")
    connection = sqlite3.connect(tmp_path / "jobs.sqlite3")
    try:
        checkpoint_count = connection.execute(
            "SELECT COUNT(*) FROM checkpoints"
        ).fetchone()[0]
        assert checkpoint_count == 0
    finally:
        connection.close()


def test_commit_transcript_is_atomic_and_ready_for_translation(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(job.id, JobStatus.DOWNLOADING)
    store.update_status(job.id, JobStatus.READY_OFFLINE)
    store.update_status(
        job.id,
        JobStatus.TRANSCRIBING,
        stage=JobStage.ASR,
        progress_permille=300,
    )

    committed = store.commit_transcript(
        job.id,
        _transcription_result(),
        artifact_path=tmp_path / "source-transcript.json",
        artifact_sha256="a" * 64,
        expected_status=JobStatus.TRANSCRIBING,
    )

    assert committed.status is JobStatus.READY_TRANSLATION
    assert committed.stage is JobStage.TRANSLATION
    assert committed.progress_permille == 450
    assert committed.details["source_language_detected"] == "en"
    assert committed.details["source_transcript_segment_count"] == 2
    segments = store.list_transcript_segments(job.id)
    assert [(item.ordinal, item.start_us, item.end_us, item.text) for item in segments] == [
        (0, 100_000, 1_250_000, "Hello world"),
        (1, 1_250_000, 2_900_000, "Second line"),
    ]
    assert segments[0].average_log_probability == pytest.approx(-0.3)
    checkpoint = store.get_checkpoint(job.id, JobStage.ASR)
    assert checkpoint is not None
    assert checkpoint.payload["artifact_sha256"] == "a" * 64
    assert checkpoint.payload["completed"] is True
    assert [event.event_type for event in store.list_events(job.id)][-2:] == [
        "job.checkpoint",
        "job.status",
    ]


def test_commit_transcript_rejects_invalid_digest_without_partial_rows(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(job.id, JobStatus.DOWNLOADING)
    store.update_status(job.id, JobStatus.READY_OFFLINE)
    store.update_status(job.id, JobStatus.TRANSCRIBING, stage=JobStage.ASR)

    with pytest.raises(ValueError, match="SHA-256"):
        store.commit_transcript(
            job.id,
            _transcription_result(),
            artifact_path=tmp_path / "source-transcript.json",
            artifact_sha256="not-a-digest",
            expected_status=JobStatus.TRANSCRIBING,
        )

    assert store.list_transcript_segments(job.id) == []
    assert store.get_checkpoint(job.id, JobStage.ASR) is None
    assert store.get_job(job.id).status is JobStatus.TRANSCRIBING


def test_transcript_rows_are_deleted_with_job(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = StateStore(database)
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(job.id, JobStatus.DOWNLOADING)
    store.update_status(job.id, JobStatus.READY_OFFLINE)
    store.update_status(job.id, JobStatus.SUBTITLE_SELECTED, stage=JobStage.SUBTITLE)
    subtitle_result = TranscriptionResult(
        source="subtitle",
        language="ja",
        language_probability=1.0,
        duration_us=1_000_000,
        segments=(TranscriptSegment(0, 900_000, "こんにちは"),),
    )
    store.commit_transcript(
        job.id,
        subtitle_result,
        artifact_path=tmp_path / "source-transcript.json",
        artifact_sha256="b" * 64,
        expected_status=JobStatus.SUBTITLE_SELECTED,
    )

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("DELETE FROM jobs WHERE id = ?", (job.id,))
        connection.commit()
        count = connection.execute(
            "SELECT COUNT(*) FROM transcript_segments WHERE job_id = ?", (job.id,)
        ).fetchone()[0]
    finally:
        connection.close()
    assert count == 0


def test_list_jobs_filters_status_and_caps_limit(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    first = store.create_job("release-1", {"rights_confirmed": True}, job_id="a")
    store.update_status(first.id, JobStatus.FAILED, retryable=True)
    second = store.create_job("release-2", {"rights_confirmed": True}, job_id="b")
    store.update_status(second.id, JobStatus.DOWNLOADING)

    active = store.list_jobs((JobStatus.DOWNLOADING, JobStatus.SUBTITLE_MATCHING), limit=10)
    assert [job.id for job in active] == ["b"]
    assert len(store.list_jobs(limit=1)) == 1


def test_database_enforces_single_active_job_atomically(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    store.create_job("release-1", {"rights_confirmed": True})

    with pytest.raises(ActiveJobExists):
        store.create_job("release-2", {"rights_confirmed": True})


def test_reopening_database_rebuilds_active_index_predicate(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    StateStore(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP INDEX jobs_one_active_idx")
        connection.execute(
            "CREATE UNIQUE INDEX jobs_one_active_idx ON jobs((1)) "
            "WHERE status IN ('created')"
        )
        connection.commit()
    finally:
        connection.close()

    reopened = StateStore(database)
    first = reopened.create_job("release-1", {"rights_confirmed": True})
    reopened.request_cancel(first.id)
    with pytest.raises(ActiveJobExists):
        reopened.create_job("release-2", {"rights_confirmed": True})

    connection = sqlite3.connect(database)
    try:
        sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'jobs_one_active_idx'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert "active_slot = 1" in sql


def test_interrupted_active_slot_migration_is_backfilled(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = StateStore(database)
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(job.id, JobStatus.DOWNLOADING)
    connection = sqlite3.connect(database)
    try:
        connection.execute("UPDATE jobs SET active_slot = 0 WHERE id = ?", (job.id,))
        connection.execute(
            "UPDATE schema_metadata SET value = '2' WHERE key = 'schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    reopened = StateStore(database)

    assert reopened.get_job(job.id).active_slot is True
    with pytest.raises(ActiveJobExists):
        reopened.create_job("release-2", {"rights_confirmed": True})


def test_schema_six_makes_legacy_track_layout_failure_retryable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = StateStore(database)
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(
        job.id,
        JobStatus.MUXING,
        stage=JobStage.EXPORT,
        force=True,
    )
    store.save_checkpoint(
        job.id,
        JobStage.TIMING,
        {"completed": True, "timeline_sha256": "a" * 64},
    )
    store.update_status(
        job.id,
        JobStatus.FAILED,
        error_code="output_track_layout_invalid",
        error_message="Video có track timecode phụ",
        retryable=False,
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE schema_metadata SET value = '5' WHERE key = 'schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    reopened = StateStore(database)
    migrated = reopened.get_job(job.id)

    assert migrated.retryable is True
    assert migrated.revision == 3
    assert reopened.list_events(job.id)[-1].event_type == "job.error_reclassified"
    resumed = reopened.resume(job.id)
    assert resumed.status is JobStatus.MUXING
    assert resumed.error_code is None
    assert reopened.get_checkpoint(job.id, JobStage.TIMING).payload["completed"] is True


def test_schema_seven_makes_legacy_duration_failure_retryable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = StateStore(database)
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(
        job.id,
        JobStatus.MUXING,
        stage=JobStage.EXPORT,
        force=True,
    )
    store.save_checkpoint(
        job.id,
        JobStage.TIMING,
        {"completed": True, "timeline_sha256": "b" * 64},
    )
    store.update_status(
        job.id,
        JobStatus.FAILED,
        error_code="output_duration_mismatch",
        error_message="Container dài hơn luồng hình",
        retryable=False,
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE schema_metadata SET value = '6' WHERE key = 'schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    reopened = StateStore(database)
    migrated = reopened.get_job(job.id)

    assert migrated.retryable is True
    assert migrated.revision == 3
    event = reopened.list_events(job.id)[-1]
    assert event.event_type == "job.error_reclassified"
    assert event.payload["reason"] == "video_canonical_duration_fix"
    resumed = reopened.resume(job.id)
    assert resumed.status is JobStatus.MUXING
    assert resumed.error_code is None
    assert reopened.get_checkpoint(job.id, JobStage.TIMING).payload["completed"] is True


def test_schema_eight_makes_cover_layout_failure_retryable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = StateStore(database)
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(
        job.id,
        JobStatus.MUXING,
        stage=JobStage.EXPORT,
        force=True,
    )
    store.save_checkpoint(
        job.id,
        JobStage.TIMING,
        {"completed": True, "timeline_sha256": "c" * 64},
    )
    store.update_status(
        job.id,
        JobStatus.FAILED,
        error_code="output_track_layout_invalid",
        error_message="Luồng ảnh bìa bị chọn thay cho video",
        retryable=False,
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE schema_metadata SET value = '7' WHERE key = 'schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    reopened = StateStore(database)
    migrated = reopened.get_job(job.id)

    assert migrated.retryable is True
    event = reopened.list_events(job.id)[-1]
    assert event.event_type == "job.error_reclassified"
    assert event.payload["reason"] == "mp4_cover_stream_selection_fix"
    resumed = reopened.resume(job.id)
    assert resumed.status is JobStatus.MUXING
    assert reopened.get_checkpoint(job.id, JobStage.TIMING).payload["completed"] is True


def test_schema_nine_makes_legacy_timing_rewrite_failure_retryable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = StateStore(database)
    job = store.create_job(
        "release-1",
        {"rights_confirmed": True, "timing_profile": "natural"},
    )
    store.update_status(
        job.id,
        JobStatus.TIMING,
        stage=JobStage.TIMING,
        force=True,
    )
    store.save_checkpoint(
        job.id,
        JobStage.TTS,
        {"completed": True, "blocks": [{"ordinal": 87}]},
    )
    store.update_status(
        job.id,
        JobStatus.FAILED,
        error_code="timing_rewrite_required",
        error_message="Khối 88 cần rút gọn",
        retryable=False,
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE schema_metadata SET value = '8' WHERE key = 'schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    reopened = StateStore(database)
    migrated = reopened.get_job(job.id)

    assert migrated.retryable is True
    event = reopened.list_events(job.id)[-1]
    assert event.event_type == "job.error_reclassified"
    assert event.payload == {
        "code": "timing_rewrite_required",
        "retryable": True,
        "reason": "timing_narration_auto_rewrite",
    }
    resumed = reopened.resume(job.id)
    assert resumed.status is JobStatus.TIMING
    assert reopened.get_checkpoint(job.id, JobStage.TTS).payload["blocks"] == [
        {"ordinal": 87}
    ]


def test_schema_ten_makes_exhausted_timing_rewrite_retryable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = StateStore(database)
    job = store.create_job(
        "release-1",
        {"rights_confirmed": True, "timing_profile": "natural"},
    )
    store.update_status(
        job.id,
        JobStatus.TIMING,
        stage=JobStage.TIMING,
        force=True,
    )
    tts_checkpoint = {
        "completed": True,
        "blocks": [{"ordinal": 180, "duration_us": 4_200_000}],
        "timing_rewrites": [
            {
                "ordinal": 180,
                "attempt": 3,
                "text": "Bản rút gọn cũ",
                "observed_duration_us": 4_200_000,
                "prompt_version": "timing-rewrite-v1",
            }
        ],
    }
    store.save_checkpoint(job.id, JobStage.TTS, tts_checkpoint)
    store.update_status(
        job.id,
        JobStatus.FAILED,
        error_code="timing_rewrite_exhausted",
        error_message="Khối 181 vẫn quá dài sau 3 lần tự rút gọn",
        retryable=False,
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE schema_metadata SET value = '9' WHERE key = 'schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    reopened = StateStore(database)
    migrated = reopened.get_job(job.id)

    assert migrated.retryable is True
    event = reopened.list_events(job.id)[-1]
    assert event.event_type == "job.error_reclassified"
    assert event.payload == {
        "code": "timing_rewrite_exhausted",
        "retryable": True,
        "reason": "timing_narration_adaptive_rewrite",
    }
    resumed = reopened.resume(job.id)
    assert resumed.status is JobStatus.TIMING
    assert reopened.get_checkpoint(job.id, JobStage.TTS).payload == tts_checkpoint


def test_schema_eleven_makes_impossible_semantic_budget_retryable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = StateStore(database)
    job = store.create_job(
        "release-1",
        {"rights_confirmed": True, "timing_profile": "natural"},
    )
    store.update_status(
        job.id,
        JobStatus.TIMING,
        stage=JobStage.TIMING,
        force=True,
    )
    tts_checkpoint = {
        "completed": True,
        "blocks": [{"ordinal": 48, "duration_us": 4_900_000}],
        "timing_rewrites": [
            {
                "ordinal": 48,
                "adaptive_attempt": 3,
                "text": "Bản rút gọn thích ứng cuối",
                "observed_duration_us": 4_900_000,
                "prompt_version": "timing-rewrite-v2",
            }
        ],
    }
    timing_checkpoint = {
        "completed": False,
        "failed_ordinal": 48,
        "available_duration_us": 3_600_000,
    }
    store.save_checkpoint(job.id, JobStage.TTS, tts_checkpoint)
    store.save_checkpoint(job.id, JobStage.TIMING, timing_checkpoint)
    store.update_status(
        job.id,
        JobStatus.FAILED,
        error_code="timing_semantic_budget_impossible",
        error_message="Khối 49 vẫn quá dài sau 3 lần rút gọn thích ứng",
        retryable=False,
    )
    unaffected = store.create_job(
        "release-1",
        {"rights_confirmed": True, "timing_profile": "natural"},
    )
    store.update_status(
        unaffected.id,
        JobStatus.FAILED,
        error_code="native_oom",
        error_message="GPU hết bộ nhớ",
        retryable=False,
    )

    connection = sqlite3.connect(database)
    try:
        checkpoint_rows_before = connection.execute(
            "SELECT stage, payload_json, hex(CAST(payload_json AS BLOB)), updated_at "
            "FROM checkpoints "
            "WHERE job_id = ? AND stage IN (?, ?) ORDER BY stage",
            (job.id, JobStage.TIMING.value, JobStage.TTS.value),
        ).fetchall()
        connection.execute(
            "UPDATE schema_metadata SET value = '10' WHERE key = 'schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    reopened = StateStore(database)
    migrated = reopened.get_job(job.id)

    assert migrated.retryable is True
    event = reopened.list_events(job.id)[-1]
    assert event.event_type == "job.error_reclassified"
    assert event.payload == {
        "code": "timing_semantic_budget_impossible",
        "retryable": True,
        "reason": "timing_narration_slack_group_fallback",
    }
    assert reopened.get_job(unaffected.id).retryable is False
    assert all(
        event.event_type != "job.error_reclassified"
        for event in reopened.list_events(unaffected.id)
    )

    connection = sqlite3.connect(database)
    try:
        checkpoint_rows_after = connection.execute(
            "SELECT stage, payload_json, hex(CAST(payload_json AS BLOB)), updated_at "
            "FROM checkpoints "
            "WHERE job_id = ? AND stage IN (?, ?) ORDER BY stage",
            (job.id, JobStage.TIMING.value, JobStage.TTS.value),
        ).fetchall()
        revision_after_first_open = connection.execute(
            "SELECT revision FROM jobs WHERE id = ?", (job.id,)
        ).fetchone()[0]
        event_count_after_first_open = connection.execute(
            "SELECT COUNT(*) FROM job_events "
            "WHERE job_id = ? AND event_type = 'job.error_reclassified'",
            (job.id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert checkpoint_rows_after == checkpoint_rows_before

    reopened_again = StateStore(database)
    connection = sqlite3.connect(database)
    try:
        assert connection.execute(
            "SELECT revision FROM jobs WHERE id = ?", (job.id,)
        ).fetchone()[0] == revision_after_first_open
        assert connection.execute(
            "SELECT COUNT(*) FROM job_events "
            "WHERE job_id = ? AND event_type = 'job.error_reclassified'",
            (job.id,),
        ).fetchone()[0] == event_count_after_first_open
    finally:
        connection.close()

    resumed = reopened_again.resume(job.id)
    assert resumed.status is JobStatus.TIMING
    assert reopened_again.get_checkpoint(job.id, JobStage.TTS).payload == tts_checkpoint
    assert (
        reopened_again.get_checkpoint(job.id, JobStage.TIMING).payload
        == timing_checkpoint
    )


def test_schema_twelve_reopens_only_valid_single_block_timing_failures(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = StateStore(database)

    def failed_job(
        release_id: str,
        timing_failure: object,
    ):
        job = store.create_job(
            release_id,
            {"rights_confirmed": True, "timing_profile": "natural"},
        )
        store.update_status(
            job.id,
            JobStatus.TIMING,
            stage=JobStage.TIMING,
            force=True,
        )
        store.update_progress(
            job.id,
            850,
            details={"timing_failure": timing_failure},
        )
        return store.update_status(
            job.id,
            JobStatus.FAILED,
            error_code="timing_group_budget_impossible",
            error_message="Ngân sách thời lượng không thể đáp ứng",
            retryable=False,
        )

    single_window = failed_job(
        "single-window",
        {
            "ordinal": 259,
            "failure_kind": "single_window_capacity",
            "failure_ordinal": 259,
            "critical_group_start_ordinal": 259,
            "critical_group_end_ordinal": 259,
            "schedule_deficit_us": 400_000,
            "rewrite_candidates": [
                {
                    "ordinal": 259,
                    "required_duration_us": 3_000_000,
                    "target_available_duration_us": 2_600_000,
                    "work_duration_us": 3_600_000,
                }
            ],
        },
    )
    elastic_postvalidation = failed_job(
        "elastic-postvalidation",
        {
            "ordinal": 12,
            "failure_kind": "elastic_postvalidation",
            "failure_ordinal": 12,
            "critical_group_start_ordinal": 12,
            "critical_group_end_ordinal": 12,
            "schedule_deficit_us": 250_000,
            "rewrite_candidates": [
                {
                    "ordinal": 12,
                    "required_duration_us": 2_000_000,
                    "target_available_duration_us": 1_750_000,
                    "work_duration_us": 2_400_000,
                }
            ],
        },
    )
    multi_block = failed_job(
        "multi-block",
        {
            "ordinal": 13,
            "failure_kind": "critical_chain_capacity",
            "failure_ordinal": 13,
            "critical_group_start_ordinal": 11,
            "critical_group_end_ordinal": 13,
        },
    )
    malformed = failed_job(
        "malformed",
        {
            "ordinal": True,
            "failure_kind": "single_window_capacity",
            "failure_ordinal": True,
            "critical_group_start_ordinal": True,
            "critical_group_end_ordinal": True,
        },
    )
    incomplete_single = failed_job(
        "incomplete-single",
        {
            "ordinal": 14,
            "failure_kind": "single_window_capacity",
            "failure_ordinal": 14,
            "critical_group_start_ordinal": 14,
            "critical_group_end_ordinal": 14,
            "schedule_deficit_us": 100_000,
        },
    )
    unrelated = store.create_job(
        "unrelated",
        {"rights_confirmed": True, "timing_profile": "natural"},
    )
    store.update_status(
        unrelated.id,
        JobStatus.FAILED,
        error_code="native_oom",
        error_message="GPU hết bộ nhớ",
        retryable=False,
    )
    tts_checkpoint = {
        "completed": True,
        "blocks": [{"ordinal": 259, "duration_us": 4_900_000}],
        "timing_rewrites": [{"ordinal": 259, "attempt": 3}],
    }
    timing_checkpoint = {
        "completed": False,
        "failed_ordinal": 259,
        "planner_policy": "natural-silent-slack-v1",
    }
    store.save_checkpoint(single_window.id, JobStage.TTS, tts_checkpoint)
    store.save_checkpoint(single_window.id, JobStage.TIMING, timing_checkpoint)

    connection = sqlite3.connect(database)
    try:
        checkpoint_rows_before = connection.execute(
            "SELECT stage, payload_json, hex(CAST(payload_json AS BLOB)), updated_at "
            "FROM checkpoints WHERE job_id = ? AND stage IN (?, ?) ORDER BY stage",
            (
                single_window.id,
                JobStage.TIMING.value,
                JobStage.TTS.value,
            ),
        ).fetchall()
        connection.execute(
            "UPDATE schema_metadata SET value = '11' WHERE key = 'schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    reopened = StateStore(database)

    for job_id in (single_window.id, elastic_postvalidation.id):
        migrated = reopened.get_job(job_id)
        assert migrated.retryable is True
        assert migrated.status is JobStatus.FAILED
        event = reopened.list_events(job_id)[-1]
        assert event.event_type == "job.error_reclassified"
        assert event.payload == {
            "code": "timing_group_budget_impossible",
            "retryable": True,
            "reason": "timing_narration_single_block_rescue",
        }
    for job_id in (
        multi_block.id,
        malformed.id,
        incomplete_single.id,
        unrelated.id,
    ):
        assert reopened.get_job(job_id).retryable is False
        assert all(
            event.event_type != "job.error_reclassified"
            for event in reopened.list_events(job_id)
        )

    connection = sqlite3.connect(database)
    try:
        checkpoint_rows_after = connection.execute(
            "SELECT stage, payload_json, hex(CAST(payload_json AS BLOB)), updated_at "
            "FROM checkpoints WHERE job_id = ? AND stage IN (?, ?) ORDER BY stage",
            (
                single_window.id,
                JobStage.TIMING.value,
                JobStage.TTS.value,
            ),
        ).fetchall()
        revision_after_first_open = connection.execute(
            "SELECT revision FROM jobs WHERE id = ?", (single_window.id,)
        ).fetchone()[0]
        event_count_after_first_open = connection.execute(
            "SELECT COUNT(*) FROM job_events "
            "WHERE job_id = ? AND event_type = 'job.error_reclassified'",
            (single_window.id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert checkpoint_rows_after == checkpoint_rows_before

    reopened_again = StateStore(database)
    connection = sqlite3.connect(database)
    try:
        assert connection.execute(
            "SELECT revision FROM jobs WHERE id = ?", (single_window.id,)
        ).fetchone()[0] == revision_after_first_open
        assert connection.execute(
            "SELECT COUNT(*) FROM job_events "
            "WHERE job_id = ? AND event_type = 'job.error_reclassified'",
            (single_window.id,),
        ).fetchone()[0] == event_count_after_first_open
    finally:
        connection.close()

    resumed = reopened_again.resume(single_window.id)
    assert resumed.status is JobStatus.TIMING
    assert resumed.error_code is None
    assert reopened_again.get_checkpoint(
        single_window.id, JobStage.TTS
    ).payload == tts_checkpoint
    assert reopened_again.get_checkpoint(
        single_window.id, JobStage.TIMING
    ).payload == timing_checkpoint


def test_future_schema_is_rejected_without_downgrading_metadata(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    StateStore(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE schema_metadata SET value = '13' WHERE key = 'schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StateError, match="schema mới hơn"):
        StateStore(database)

    connection = sqlite3.connect(database)
    try:
        version = connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert version == "13"


def test_cancel_from_paused_is_atomic_and_idempotent(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(job.id, JobStatus.DOWNLOADING)
    store.update_status(job.id, JobStatus.PAUSED)

    cancelling = store.request_cancel(job.id)
    assert cancelling.status == JobStatus.CANCELLING
    assert cancelling.active_slot is False
    cancelled = store.finalize_cancel(job.id)
    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.cancel_requested is True
    assert store.request_cancel(job.id) == cancelled
    assert store.list_events(job.id)[-1].payload["status"] == "cancelled"


def test_inactive_cancellation_can_coexist_with_an_active_job(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    old = store.create_job("release-old", {"rights_confirmed": True})
    store.update_status(old.id, JobStatus.DOWNLOADING)
    store.update_status(old.id, JobStatus.PAUSED)
    active = store.create_job("release-active", {"rights_confirmed": True})

    cancelling = store.request_cancel(old.id)

    assert cancelling.status is JobStatus.CANCELLING
    assert cancelling.active_slot is False
    assert store.list_active_jobs(limit=10) == [active]


def test_append_warning_preserves_status_and_emits_event(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(job.id, JobStatus.DOWNLOADING)

    warned = store.append_warning(job.id, "slow_source", "Nguồn tải đang chậm")

    assert warned.status == JobStatus.DOWNLOADING
    assert warned.details["warnings"] == [
        {"code": "slow_source", "message": "Nguồn tải đang chậm"}
    ]
    assert store.list_events(job.id)[-1].event_type == "job.warning"


def test_unchanged_progress_does_not_append_duplicate_event(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job("release-1", {"rights_confirmed": True})
    details = {"task_id": "fixture", "downloaded_bytes": 10}
    store.update_status(
        job.id,
        JobStatus.DOWNLOADING,
        progress_permille=10,
        details=details,
    )
    event_count = len(store.list_events(job.id))

    unchanged = store.update_progress(job.id, 10, details=details)

    assert unchanged.revision == 1
    assert len(store.list_events(job.id)) == event_count


def test_resume_restores_paused_or_retryable_failed_stage(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    paused = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(paused.id, JobStatus.DOWNLOADING)
    store.update_status(paused.id, JobStatus.PAUSED)
    resumed = store.resume(paused.id)
    assert resumed.status == JobStatus.DOWNLOADING
    store.update_status(paused.id, JobStatus.PAUSED)

    failed = store.create_job("release-2", {"rights_confirmed": True})
    store.update_status(
        failed.id,
        JobStatus.FAILED,
        error_code="network",
        error_message="Mất kết nối",
        retryable=True,
    )
    retried = store.resume(failed.id)
    assert retried.status == JobStatus.CREATED
    assert retried.error_code is None


def test_non_retryable_failure_and_invalid_transition_are_rejected(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(
        job.id,
        JobStatus.FAILED,
        error_code="rights",
        error_message="Không có quyền",
        retryable=False,
    )

    with pytest.raises(InvalidTransition, match="không thể thử lại"):
        store.resume(job.id)
    with pytest.raises(InvalidTransition):
        store.update_status(job.id, JobStatus.COMPLETED)


def test_run_if_job_revision_rejects_stale_snapshot(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    created = store.create_job("release-1", {"rights_confirmed": True})
    calls: list[int] = []

    assert store.run_if_job_revision(
        created.id,
        created.revision,
        lambda current: calls.append(current.revision) is None,
    ) is True
    assert calls == [created.revision]

    store.update_status(created.id, JobStatus.DOWNLOADING)
    assert store.run_if_job_revision(
        created.id,
        created.revision,
        lambda _current: calls.append(999) is None,
    ) is False
    assert calls == [created.revision]
