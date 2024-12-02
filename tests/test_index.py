from unittest.mock import Mock

import pytest

from semantic_search.index import create_documents, create_schema
from semantic_search.utils import VECTOR_DIM


@pytest.fixture
def mock_embedding_model():
    model = Mock()
    model.encode_documents.return_value = [
        [0.1] * VECTOR_DIM,
        [0.2] * VECTOR_DIM,
    ]
    return model


def test_create_schema():
    schema = create_schema()
    fields = {field.name: field for field in schema.fields}
    assert "id" in fields
    assert "title" in fields
    assert "vector" in fields
    assert "snippet" in fields
    assert "published_date" in fields


def test_create_documents(mock_embedding_model):
    valid_articles = [
        {
            "link": "http://example.com/1",
            "title": "Title 1",
            "summary": "Summary 1",
        },
        {
            "link": "http://example.com/2",
            "title": "Title 2",
            "summary": "Summary 2",
        },
    ]
    documents = create_documents(valid_articles, mock_embedding_model)

    assert len(documents) == 2
    assert documents[0]["id"] == "http://example.com/1"
    assert documents[0]["title"] == "Title 1"
    assert documents[0]["snippet"] == "Summary 1"
    assert len(documents[0]["vector"]) == VECTOR_DIM


def test_create_documents_missing_fields(mock_embedding_model):
    invalid_articles = [
        {
            "title": "Test Article 1",
            "summary": "This is a test summary 1",
        },
        {
            "link": "http://example.com/2",
            "summary": "This is a test summary 2",
        },
        {
            "link": "http://example.com/3",
            "title": "Test Article 3",
        },
    ]

    with pytest.raises(KeyError):
        create_documents(invalid_articles, mock_embedding_model)


def test_create_documents_empty_articles(mock_embedding_model):
    empty_articles = []
    documents = create_documents(empty_articles, mock_embedding_model)

    assert len(documents) == 0


def test_create_documents_none_articles(mock_embedding_model):
    with pytest.raises(TypeError):
        create_documents(None, mock_embedding_model)
