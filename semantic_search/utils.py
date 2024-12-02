from os import path
from milvus_model.dense import SentenceTransformerEmbeddingFunction
from typing import Dict

DB_NAME = "people_news"

# DATA PATHS
DATA_DIR = "data"
RSS_FEED_RESOURCES_DIR = path.join(DATA_DIR, "rss_feed_resources")
ARTICLES_DIR = path.join(DATA_DIR, "articles")


# SEARCH CONFIG
METRIC_TYPE: str = "COSINE"
INDEX_TYPE: str = "HNSW"
VECTOR_DIM: int = 384
MODEL_NAME: str = "all-MiniLM-L6-v2"
DEVICE: str = "cpu"

def model(model_name: str = MODEL_NAME, device: str = DEVICE) -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(
        model_name=model_name, device=device
    )


def search_params(metric_type: str = METRIC_TYPE) -> Dict[str, str]:
    return {"metric_type": metric_type}
