from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dub_server.asr import (
    FasterWhisperRecognizer,
    LanguageDetectionRequired,
    NoSpeechError,
    TranscriptionError,
    normalize_segments,
    normalize_whisper_language,
)


class FakeModel:
    def __init__(self, segments: list[object], info: object) -> None:
        self.segments = segments
        self.info = info
        self.calls: list[tuple[str, dict[str, object]]] = []

    def transcribe(self, source: str, **kwargs: object) -> tuple[list[object], object]:
        self.calls.append((source, kwargs))
        return self.segments, self.info


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"fixture")
    model = tmp_path / "model"
    model.mkdir()
    return media, model


def test_recognizer_uses_verified_local_path_and_cuda(tmp_path: Path) -> None:
    media, model_path = _paths(tmp_path)
    fake = FakeModel(
        [
            SimpleNamespace(
                start=-1.0,
                end=1.25,
                text="  Hello   world ",
                avg_logprob=-0.42,
                no_speech_prob=0.08,
            ),
            SimpleNamespace(
                start=1.0,
                end=2.0,
                text="again",
                avg_logprob=-0.21,
                no_speech_prob=0.02,
            ),
        ],
        SimpleNamespace(
            language="en",
            language_probability=0.93,
            all_language_probs=[("en", 0.93), ("de", 0.04)],
        ),
    )
    factory_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def factory(*args: object, **kwargs: object) -> FakeModel:
        factory_calls.append((args, kwargs))
        return fake

    progress: list[tuple[int, int]] = []
    result = FasterWhisperRecognizer(model_factory=factory).transcribe(
        media,
        model_path=model_path,
        model_id="asr-large-v3",
        compute_type="float16",
        language="auto",
        duration_us=2_000_000,
        on_progress=lambda end_us, count: progress.append((end_us, count)),
    )

    assert factory_calls == [
        (
            (str(model_path.resolve()),),
            {
                "device": "cuda",
                "compute_type": "float16",
                "local_files_only": True,
            },
        )
    ]
    assert fake.calls[0][0] == str(media.resolve())
    assert fake.calls[0][1]["task"] == "transcribe"
    assert result.language == "en"
    assert result.model_id == "asr-large-v3"
    assert [(item.start_us, item.end_us) for item in result.segments] == [
        (0, 1_250_000),
        (1_250_000, 2_000_000),
    ]
    assert result.segments[0].average_log_probability == pytest.approx(-0.42)
    assert result.segments[0].no_speech_probability == pytest.approx(0.08)
    assert progress == [(1_250_000, 1), (2_000_000, 2)]


def test_recognizer_supports_cpu_int8(tmp_path: Path) -> None:
    media, model_path = _paths(tmp_path)
    fake = FakeModel(
        [SimpleNamespace(start=0.0, end=1.0, text="hello")],
        SimpleNamespace(language="en", language_probability=1.0),
    )
    calls: list[dict[str, object]] = []

    def factory(*_args: object, **kwargs: object) -> FakeModel:
        calls.append(kwargs)
        return fake

    FasterWhisperRecognizer(model_factory=factory, device="cpu").transcribe(
        media,
        model_path=model_path,
        model_id="asr-small",
        compute_type="int8",
        language="en",
        duration_us=1_000_000,
    )

    assert calls[0]["device"] == "cpu"
    assert calls[0]["compute_type"] == "int8"


def test_auto_language_below_threshold_requires_user_selection(tmp_path: Path) -> None:
    media, model_path = _paths(tmp_path)
    fake = FakeModel(
        [SimpleNamespace(start=0, end=1, text="hello")],
        SimpleNamespace(
            language="en",
            language_probability=0.49,
            all_language_probs=[("en", 0.49), ("ja", 0.31)],
        ),
    )

    with pytest.raises(LanguageDetectionRequired) as captured:
        FasterWhisperRecognizer(model_factory=lambda *args, **kwargs: fake).transcribe(
            media,
            model_path=model_path,
            model_id="asr-small",
            compute_type="float16",
            language=None,
            duration_us=2_000_000,
        )

    assert captured.value.detected_language == "en"
    assert captured.value.probability == pytest.approx(0.49)
    assert captured.value.alternatives == (("en", 0.49), ("ja", 0.31))


def test_explicit_language_bypasses_auto_confidence_gate(tmp_path: Path) -> None:
    media, model_path = _paths(tmp_path)
    fake = FakeModel(
        [SimpleNamespace(start=0, end=1, text="xin chao")],
        SimpleNamespace(language="vi", language_probability=0.1),
    )

    result = FasterWhisperRecognizer(model_factory=lambda *args, **kwargs: fake).transcribe(
        media,
        model_path=model_path,
        model_id="asr-small",
        compute_type="int8_float16",
        language="vie-VN",
        duration_us=2_000_000,
    )

    assert result.language == "vi"
    assert fake.calls[0][1]["language"] == "vi"


def test_empty_or_invalid_segments_report_no_speech(tmp_path: Path) -> None:
    media, model_path = _paths(tmp_path)
    fake = FakeModel(
        [
            SimpleNamespace(start=0, end=0, text="zero"),
            SimpleNamespace(start=0, end=1, text="   "),
        ],
        SimpleNamespace(language="en", language_probability=0.9),
    )

    with pytest.raises(NoSpeechError):
        FasterWhisperRecognizer(model_factory=lambda *args, **kwargs: fake).transcribe(
            media,
            model_path=model_path,
            model_id="asr-small",
            compute_type="float16",
            language=None,
            duration_us=2_000_000,
        )


def test_model_must_be_an_installed_directory(tmp_path: Path) -> None:
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"fixture")
    with pytest.raises(TranscriptionError) as captured:
        FasterWhisperRecognizer(model_factory=lambda *args, **kwargs: None).transcribe(
            media,
            model_path=tmp_path / "missing-model",
            model_id="asr-small",
            compute_type="float16",
            language=None,
            duration_us=2_000_000,
        )
    assert captured.value.code == "model_missing"


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [(None, None), ("auto", None), ("eng-US", "en"), ("jpn", "ja"), ("VI_vn", "vi")],
)
def test_language_normalization(input_value: str | None, expected: str | None) -> None:
    assert normalize_whisper_language(input_value) == expected


def test_segment_normalization_drops_invalid_and_clamps_duration() -> None:
    segments = normalize_segments(
        [
            SimpleNamespace(start=float("nan"), end=1, text="bad"),
            SimpleNamespace(start=0.25, end=99, text="kept"),
            SimpleNamespace(start=2, end=3, text="after end"),
        ],
        duration_us=1_000_000,
    )
    assert [(item.start_us, item.end_us, item.text) for item in segments] == [
        (250_000, 1_000_000, "kept")
    ]
