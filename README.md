# Semantic Search Implementation

This project implements a semantic search system using Milvus vector database and sentence transformers. 

## Components

### 1. Crawler (`crawler.py`)
- Fetches articles from specified RSS feeds
- Saves articles in JSONL format

### 2. Indexing System (`semantic_search/index.py`)
- Processes articles from JSONL files
- Creates embeddings using sentence transformers
- Inserts articles into the Milvus vector database

### 3. Search Interface (`semantic_search/search.py`)
- Provides a Gradio-based web interface for searching
- Converts search queries into vector embeddings
- Performs similarity search using cosine distance
- Returns top-k most similar articles with their metadata

## Usage

### Crawling Articles
```bash
python semantic_search/crawl.py --feed-filename <feed_filename>
```

### Indexing Articles
```bash
python semantic_search/index.py --db-name <db_name> --articles-dir <articles_dir> --recreate <recreate>
```

### Running the Search Interface
```bash
python -m semantic_search.search
```

This will launch a web interface where you can:
- Enter natural language queries
- Adjust the number of results (1-20)
- View matching articles with their titles, snippets, and similarity scores

Runs on local URL: http://127.0.0.1:7860

## Pipeline Flow

1. Crawler fetches articles from RSS feeds → JSONL files
2. Indexer processes JSONL files → Vector database
3. Search interface queries vector database → User results

## Technical Details

- **Vector Database**: Milvus
- **Embedding Model**: Sentence Transformer (configured in utils)
- **Vector Dimension**: 384 (using MiniLM-L6-v2 by default)
- **Similarity Metric**: Cosine similarity
- **Index Type**: HNSW (Hierarchical Navigable Small World)

## Notes

### Design Choices Explained

- **The use of a Vector DB**: I preferred not to write "from scratch" code to mock a vector db but rather to use an open-source one for several reasons:
1. simplicity (4-hour exercise)
2. the fact that it already has integrated algorithms to accelerate search for each query (ANN search)
- **Gradio**: Provides a simple web interface for interacting with the search system.
- **MiniLM-L6-v2**: I chose this model because it's a lightweight model that's good for semantic similarity tasks + because it's the default model used in the Milvus documentation.