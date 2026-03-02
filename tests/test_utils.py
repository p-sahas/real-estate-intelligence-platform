from context_engineering.domain import utils
import os
import sys
import pytest

# make sure src directory is on path for imports
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")))


class DummyDoc:
    def __init__(self, content, url=None, title=None, strategy=None):
        self.page_content = content
        self.metadata = {}
        if url:
            self.metadata["url"] = url
        if title:
            self.metadata["title"] = title
        if strategy:
            self.metadata["strategy"] = strategy


def test_format_docs_basic():
    docs = [
        DummyDoc("hello world", url="http://a", title="A"),
        DummyDoc("more content", url="http://b", title="B"),
    ]

    out = utils.format_docs(docs)
    assert "Source 1" in out
    assert "Source 2" in out
    assert "hello world" in out
    assert "more content" in out


def test_format_docs_missing_metadata():
    # should default to N/A when metadata keys missing
    docs = [DummyDoc("x")]
    out = utils.format_docs(docs)
    assert "N/A" in out


def test_calculate_confidence_empty():
    assert utils.calculate_confidence([], "anything") == 0.0


def test_calculate_confidence_full_overlap():
    # provide a document that contains the query words and has different strategies
    docs = [
        DummyDoc("hello world foo bar", strategy="s1"),
        DummyDoc("hello world and more", strategy="s2"),
    ]
    # query two words
    score = utils.calculate_confidence(docs, "hello world")
    assert 0.0 <= score <= 1.0
    # since both docs contain both words, keyword_score should be 1
    # diversity_score should be 2/3
    # length_score will be >0
    assert score > 0.5


def test_extract_citations():
    text = "This answer refers to [http://example.com] and [not a url] and [https://site.com/page]"
    urls = utils.extract_citations(text)
    assert "http://example.com" in urls
    assert "https://site.com/page" in urls
    assert "not a url" not in urls


def test_truncate_text():
    short = "short text"
    assert utils.truncate_text(short, max_length=50) == short

    long = "word " * 100
    truncated = utils.truncate_text(long, max_length=20)
    assert len(truncated) <= 23  # allow ellipsis
    assert truncated.endswith("...")


@pytest.mark.parametrize("text,maxlen,expected", [
    ("a b c", 5, "a b c"),
    ("one two three four", 10, "one two...")
])
def test_truncate_various(text, maxlen, expected):
    assert utils.truncate_text(text, maxlen) == expected
