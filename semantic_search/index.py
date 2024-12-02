import json
import os
from typing import Any, Dict, List

import click
from loguru import logger
from milvus_model.dense import SentenceTransformerEmbeddingFunction
from pymilvus import CollectionSchema, FieldSchema, MilvusClient
from pymilvus.orm.types import DataType

from semantic_search.utils import ARTICLES_DIR, DB_NAME, METRIC_TYPE, INDEX_TYPE, VECTOR_DIM, model

REQUIRED_FIELDS = ["link", "title", "summary"]


def load_articles(articles_dir: str) -> List[Dict[str, Any]]:
    """
    Load articles from a directory of JSONL files
    """
    articles = []
    for filename in os.listdir(articles_dir):
        if filename.endswith(".jsonl"):
            with open(os.path.join(articles_dir, filename), "r") as file:
                for line in file:
                    article = json.loads(line)
                    articles.append(article)
    return articles


def create_collection(
    client: MilvusClient,
    collection_name: str,
    dimension: int = VECTOR_DIM,
    recreate: bool = False,
    collection_args: Dict[str, Any] = {},
) -> None:
    """
    Creates a new collection, dropping the existing one if it exists and recreates if True
    Args:
        client (MilvusClient): MilvusClient instance
        collection_name (str): name of the collection
        dimension (int, optional): dimension of the vectors (default 384 for MiniLM-L6-v2)
        recreate (bool, optional): whether to recreate the collection if it already exists
        collection_args (Dict[str, Any], optional): additional keyword arguments to pass to `client.create_collection()`
    """
    if client.has_collection(collection_name=collection_name):
        if recreate:
            logger.info(f"Dropping existing collection {collection_name}")
            client.drop_collection(collection_name=collection_name)
        else:
            return
    logger.info(f"Creating collection {collection_name}")
    client.create_collection(
        collection_name=collection_name, dimension=dimension, **collection_args
    )


def create_schema() -> CollectionSchema:
    """
    Create the index schema
    To be improved -> schema in a json file as input to create FieldSchema objects

    Returns:
        CollectionSchema: the schema for the people_news collection
    """
    id_field = FieldSchema(
        name="id",
        dtype=DataType.VARCHAR,
        is_primary=True,
    )
    title_field = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000)
    vector_field = FieldSchema(
        name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM
    )
    snippet = FieldSchema(name="snippet", dtype=DataType.VARCHAR, max_length=10000)
    published_date = FieldSchema(name="published_date", dtype=DataType.FLOAT)

    schema = CollectionSchema(
        fields=[id_field, vector_field, snippet, published_date, title_field],
        description=f"{DB_NAME}_schema",
    )
    schema.verify()

    return schema


def create_documents(
    articles: List[Dict[str, Any]], model: SentenceTransformerEmbeddingFunction
) -> List[Dict[str, Any]]:
    """
    Create documents from articles
    Raises:
        KeyError: if an article is missing required fields
    Args:
        articles (List[Dict[str, Any]]): list of articles
        model (SentenceTransformerEmbeddingFunction): embedding model
    Returns:
        List[Dict[str, Any]]: list of documents
    """
    documents = []
    for i, article in enumerate(articles):
        document = {}
        # Validate required fields
        if not all(key in article for key in REQUIRED_FIELDS):
            missing_keys = [key for key in REQUIRED_FIELDS if key not in article]
            raise KeyError(
                f"Article at line {i+1} is missing required fields: {missing_keys}"
            )

        document["id"] = article["link"]
        document["title"] = article["title"]
        document["snippet"] = article["summary"]
        document["vector"] = model.encode_documents([document["title"]])[
            0
        ]  # to be improved: batch process embeddings
        documents.append(document)
    return documents


@click.command()
@click.option(
    "--db-name", default=DB_NAME, help=f"Name of the database, defaults to {DB_NAME}"
)
@click.option(
    "--articles-dir",
    default=ARTICLES_DIR,
    help=f"Directory containing article files created by the crawler, defaults to {ARTICLES_DIR}",
)
@click.option(
    "--recreate/--no-recreate",
    default=True,
    help="Whether to recreate the collection if it exists",
)
def main(db_name: str, articles_dir: str, recreate: bool):
    """Initialize and populate a Milvus vector database with article embeddings."""
    client = MilvusClient(f"{db_name}.db")

    create_collection(
        client,
        db_name,
        recreate=recreate,
        collection_args={"id_type": "string", "max_length": 10000},
    )
    create_schema()

    articles = load_articles(articles_dir=articles_dir)
    documents = create_documents(articles=articles, model=model())
    client.insert(collection_name=db_name, data=documents)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type=METRIC_TYPE,
        index_type=INDEX_TYPE,
        index_name=db_name,
    )
    logger.info(f"Creating index {db_name} with params {index_params}")
    client.create_index(
        collection_name=db_name,
        index_params=index_params,
    )


if __name__ == "__main__":
    main()
