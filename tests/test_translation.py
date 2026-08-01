from __future__ import annotations

import socket
import unicodedata
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from dub_server.translation import (
    TranslationBlock,
    TranslationError,
    build_translation_blocks,
    bypass_vietnamese_translation,
    translate_blocks,
)


def _segment(start_us: int, end_us: int, text: str, ordinal: int):
    return SimpleNamespace(
        start_us=start_us,
        end_us=end_us,
        text=text,
        ordinal=ordinal,
    )


def _word_tokens(text: str) -> int:
    return len(text.split())


class FakeTranslator:
    def __init__(self, responses: Sequence[Sequence[str]]) -> None:
        self._responses = [list(response) for response in responses]
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def count_tokens(self, text: str) -> int:
        return _word_tokens(text)

    def translate_batch(
        self,
        texts: Sequence[str],
        *,
        source_language: str,
        target_language: str = "vi",
        on_progress=None,
    ) -> list[str]:
        self.calls.append(
            {
                "texts": list(texts),
                "source_language": source_language,
                "target_language": target_language,
                "on_progress": on_progress,
            }
        )
        if not self._responses:
            raise AssertionError("Fake translator received an unexpected call")
        return self._responses.pop(0)

    def close(self) -> None:
        self.closed = True


def test_block_normalizes_unicode_whitespace_and_preserves_ordinals() -> None:
    block = TranslationBlock(
        100,
        200,
        "  Tie\u0302\u0301ng\n  Vie\u0323\u0302t  ",
        "  Bản   dịch ",
        (2, 3),
    )

    assert block.source_text == unicodedata.normalize("NFC", "Tiếng Việt")
    assert block.translated_text == "Bản dịch"
    assert block.source_ordinals == (2, 3)


def test_builder_clamps_overlap_and_keeps_integer_monotonic_timeline() -> None:
    blocks = build_translation_blocks(
        (
            _segment(-100_000, 600_000, "First", 0),
            _segment(500_000, 1_200_000, "Second", 1),
        ),
        duration_us=1_500_000,
        token_counter=_word_tokens,
        mandatory_merge_us=1,
        short_merge_us=1,
    )

    assert [
        (block.start_us, block.end_us, block.source_text, block.source_ordinals)
        for block in blocks
    ] == [
        (0, 600_000, "First", (0,)),
        (600_000, 1_200_000, "Second", (1,)),
    ]
    assert all(isinstance(block.start_us, int) for block in blocks)
    assert all(isinstance(block.end_us, int) for block in blocks)


def test_builder_merges_mandatory_and_nearby_short_segments() -> None:
    blocks = build_translation_blocks(
        (
            _segment(0, 200_000, "Very", 0),
            _segment(300_000, 900_000, "short", 1),
            _segment(2_000_000, 3_500_000, "Separate block", 2),
        ),
        duration_us=4_000_000,
        token_counter=_word_tokens,
    )

    assert blocks == (
        TranslationBlock(0, 900_000, "Very short", source_ordinals=(0, 1)),
        TranslationBlock(
            2_000_000,
            3_500_000,
            "Separate block",
            source_ordinals=(2,),
        ),
    )


def test_builder_splits_at_text_boundaries_to_duration_and_token_limits() -> None:
    source_text = "one two three. four five six."
    blocks = build_translation_blocks(
        (_segment(0, 10_000_000, source_text, 7),),
        duration_us=10_000_000,
        token_counter=_word_tokens,
        max_duration_us=6_000_000,
        max_tokens=3,
        mandatory_merge_us=1,
        short_merge_us=1,
    )

    assert len(blocks) >= 2
    assert blocks[0].start_us == 0
    assert blocks[-1].end_us == 10_000_000
    assert all(left.end_us == right.start_us for left, right in zip(blocks, blocks[1:]))
    assert all(block.end_us - block.start_us <= 6_000_000 for block in blocks)
    assert all(_word_tokens(block.source_text) <= 3 for block in blocks)
    assert all(block.source_ordinals == (7,) for block in blocks)
    assert " ".join(block.source_text for block in blocks) == source_text


def test_builder_rejects_invalid_token_counter_and_unmergeable_tiny_segment() -> None:
    with pytest.raises(TranslationError) as invalid_counter:
        build_translation_blocks(
            (_segment(0, 1_000_000, "content", 0),),
            duration_us=1_000_000,
            token_counter=lambda _text: 0,
        )
    assert invalid_counter.value.code == "translation_tokenizer_failed"
    assert invalid_counter.value.retryable is True

    with pytest.raises(TranslationError) as unmergeable:
        build_translation_blocks(
            (
                _segment(0, 100_000, "one", 0),
                _segment(200_000, 1_200_000, "two", 1),
            ),
            duration_us=1_500_000,
            token_counter=_word_tokens,
            max_tokens=1,
        )
    assert unmergeable.value.code == "translation_short_block_unmergeable"
    assert unmergeable.value.retryable is False


def test_translate_batch_targets_vietnamese_and_normalizes_outputs() -> None:
    blocks = (
        TranslationBlock(0, 1_000_000, "Hello", source_ordinals=(0,)),
        TranslationBlock(1_500_000, 3_000_000, "How are you?", source_ordinals=(1,)),
    )
    translator = FakeTranslator([["  Xin  chào ", " Bạn khỏe không? "]])

    translated = translate_blocks(
        translator,
        blocks,
        source_language="en",
    )

    assert [block.translated_text for block in translated] == [
        "Xin chào",
        "Bạn khỏe không?",
    ]
    assert translator.calls == [
        {
            "texts": ["Hello", "How are you?"],
            "source_language": "en",
            "target_language": "vi",
            "on_progress": None,
        }
    ]


def test_empty_output_is_retried_once_by_splitting_only_that_block() -> None:
    original = TranslationBlock(
        100_000,
        4_100_000,
        "First sentence. Second sentence.",
        source_ordinals=(4, 5),
    )
    translator = FakeTranslator([["   "], ["Câu thứ nhất.", "Câu thứ hai."]])

    translated = translate_blocks(
        translator,
        (original,),
        source_language="en",
    )

    assert len(translator.calls) == 2
    assert translator.calls[0]["texts"] == [original.source_text]
    assert translator.calls[1]["texts"] == [
        block.source_text for block in translated
    ]
    assert translated[0].start_us == original.start_us
    assert translated[0].end_us == translated[1].start_us
    assert translated[1].end_us == original.end_us
    assert [block.translated_text for block in translated] == [
        "Câu thứ nhất.",
        "Câu thứ hai.",
    ]
    assert all(block.source_ordinals == (4, 5) for block in translated)


def test_empty_retry_failure_never_copies_source_text_into_translation() -> None:
    block = TranslationBlock(0, 2_000_000, "Original source text")
    translator = FakeTranslator([[""], ["Translated half", ""]])

    with pytest.raises(TranslationError) as captured:
        translate_blocks(translator, (block,), source_language="en")

    assert captured.value.code == "translation_output_empty"
    assert captured.value.retryable is True
    assert len(translator.calls) == 2


def test_vietnamese_bypass_preserves_timeline_without_translator() -> None:
    blocks = (
        TranslationBlock(0, 1_000_000, "Xin chào", source_ordinals=(0,)),
        TranslationBlock(2_000_000, 3_000_000, "Việt Nam", source_ordinals=(1,)),
    )

    translated = bypass_vietnamese_translation(blocks)

    assert [block.translated_text for block in translated] == [
        "Xin chào",
        "Việt Nam",
    ]
    assert [
        (block.start_us, block.end_us, block.source_ordinals)
        for block in translated
    ] == [
        (0, 1_000_000, (0,)),
        (2_000_000, 3_000_000, (1,)),
    ]


def test_translation_contracts_do_not_access_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_socket(*_args: object, **_kwargs: object) -> socket.socket:
        raise AssertionError("translation contracts must remain offline")

    monkeypatch.setattr(socket, "socket", reject_socket)
    blocks = build_translation_blocks(
        (_segment(0, 1_000_000, "Offline source", 0),),
        duration_us=1_000_000,
        token_counter=_word_tokens,
        mandatory_merge_us=1,
        short_merge_us=1,
    )
    translated = translate_blocks(
        FakeTranslator([["Bản dịch ngoại tuyến"]]),
        blocks,
        source_language="en",
    )
    assert translated[0].translated_text == "Bản dịch ngoại tuyến"
