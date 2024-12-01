from unittest.mock import Mock, patch

import numpy as np
import pytest

from semantic_search.search import search_documents
from semantic_search.utils import DB_NAME


@pytest.fixture
def mock_milvus_client():
    with patch("semantic_search.search.CLIENT") as mock_client:
        # Create sample search results
        mock_hits = [
            [
                {
                    "id": "1",
                    "entity": {
                        "title": "Test Document 1",
                        "snippet": "This is a test snippet 1",
                    },
                    "distance": 0.8,
                },
                {
                    "id": "2",
                    "entity": {
                        "title": "Test Document 2",
                        "snippet": "This is a test snippet 2",
                    },
                    "distance": 0.6,
                },
            ]
        ]
        mock_client.search.return_value = mock_hits
        yield mock_client


@pytest.fixture
def mock_embedding_model():
    with patch("semantic_search.search.embedding_model") as mock_model:
        mock_model.encode_queries.return_value = [np.array([0.1, 0.2, 0.3])]
        yield mock_model


def test_search_documents_basic(mock_milvus_client, mock_embedding_model):
    query = "test query"
    top_k = 2
    result = search_documents(query, top_k)

    # Verify Milvus client was called correctly
    call_args = mock_milvus_client.search.call_args[1]
    assert call_args["collection_name"] == DB_NAME
    assert call_args["limit"] == top_k
    assert call_args["search_params"] == {"metric_type": "COSINE"}

    # Verify output format
    assert "Result 1:" in result
    assert "Result 2:" in result
    assert "Test Document 1" in result
    assert "Test Document 2" in result
    assert "This is a test snippet 1" in result
    assert "This is a test snippet 2" in result
    assert "0.8000" in result
    assert "0.6000" in result


def test_search_documents_empty_results(mock_milvus_client, mock_embedding_model):
    mock_milvus_client.search.return_value = [[]]
    result = search_documents("query", 5)
    assert result == ""


def test_search_documents_missing_fields(mock_milvus_client):
    mock_hits = [[{"id": "1", "entity": {}, "distance": 0.8}]]
    mock_milvus_client.search.return_value = mock_hits

    result = search_documents("query", 1)

    assert "Title: " in result
    assert "Snippet: " in result
    assert "0.8000" in result
