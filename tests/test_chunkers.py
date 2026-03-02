from context_engineering.application.ingest_documents_service.chunkers import (
    ChunkingService,
    semantic_chunk,
    fixed_chunk,
    sliding_chunk,
    parent_child_chunk,
    late_chunk_index,
    late_chunk_split,
    count_tokens,
)
from context_engineering.application.ingest_documents_service import chunkers
import os
import sys
import pytest

# ensure source code is importable
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")))


SIMPLE_DOC = {"url": "u", "title": "t", "content": "Hello world"}


def test_count_tokens_basic():
    # should return integer >= number of words
    result = count_tokens("hello world")
    assert isinstance(result, int)
    assert result >= 2


def test_semantic_chunk_no_headings():
    chunks = semantic_chunk([SIMPLE_DOC])
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0]["strategy"] == "semantic"


def test_fixed_chunk_small():
    chunks = fixed_chunk([SIMPLE_DOC])
    assert chunks
    assert chunks[0]["strategy"] == "fixed"
    assert chunks[0]["url"] == "u"
    assert chunks[0]["token_count"] >= 1


def test_sliding_chunk_small():
    chunks = sliding_chunk([SIMPLE_DOC])
    assert chunks
    assert chunks[0]["strategy"] == "sliding"
    assert chunks[0]["window_index"] == 0


def test_parent_child_chunk_basic():
    children, parents = parent_child_chunk([SIMPLE_DOC])
    assert isinstance(children, list)
    assert isinstance(parents, list)
    assert parents
    for child in children:
        assert child["parent_id"].startswith("u::parent")
        assert child["strategy"] == "child"


def test_late_chunk_and_split():
    chunks = late_chunk_index([SIMPLE_DOC])
    assert chunks
    assert chunks[0]["strategy"] == "late_chunk_base"
    # splitting a passage containing none of the query returns full passage
    splits = late_chunk_split("some text here", "xyz")
    assert len(splits) == 1
    assert splits[0]["text"] == "some text here"

    # with a match
    passage = "this example contains queryterm somewhere"
    res = late_chunk_split(passage, "queryterm")
    assert any("queryterm" in c["text"] for c in res)


def test_chunking_service_strategies():
    svc = ChunkingService()
    avail = svc.available_strategies()
    assert set(avail) >= {"semantic", "fixed",
                          "sliding", "parent_child", "late_chunk"}
    with pytest.raises(ValueError):
        svc.chunk([SIMPLE_DOC], strategy="unknown")

    # ensure service returns same as direct call
    assert svc.chunk(
        [SIMPLE_DOC], strategy="fixed") == fixed_chunk([SIMPLE_DOC])


@pytest.mark.parametrize("strategy", ["semantic", "fixed", "sliding", "parent_child", "late_chunk"])
def test_service_returns_nonempty(strategy):
    svc = ChunkingService()
    result = svc.chunk([SIMPLE_DOC], strategy=strategy)
    assert result is not None
