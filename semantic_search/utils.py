from os import path
from dataclasses import dataclass
from milvus_model.dense import SentenceTransformerEmbeddingFunction


DB_NAME = "people_news"

# DATA PATHS
DATA_DIR = "data"
RSS_FEED_RESOURCES_DIR = path.join(DATA_DIR, "rss_feed_resources")
ARTICLES_DIR = path.join(DATA_DIR, "articles")


@dataclass
class SearchConfig:
    metric_type: str = "COSINE"
    index_type: str = "HNSW"
    vector_dim: int = 384
    model_name: str = "all-MiniLM-L6-v2"
    DEVICE = "cpu"

    @property
    def model(self) -> SentenceTransformerEmbeddingFunction:
        return SentenceTransformerEmbeddingFunction(
            model_name=self.model_name, device=self.DEVICE
        )

    @property
    def search_params(self):
        return {"metric_type": self.metric_type}
