import json
from unittest.mock import mock_open, patch

import pytest

from semantic_search.crawler import get_rss_urls


@pytest.fixture
def sample_rss_data():
    return [{"url": "http://example1.com/feed"}, {"url": "http://example2.com/feed"}]


def test_get_rss_urls(sample_rss_data):
    file_content = "\n".join(json.dumps(data) for data in sample_rss_data)

    # Mock the file open operation
    with patch("builtins.open", mock_open(read_data=file_content)):
        urls = get_rss_urls("test_feeds.jsonl")

        assert len(urls) == 2
        assert urls == ["http://example1.com/feed", "http://example2.com/feed"]
