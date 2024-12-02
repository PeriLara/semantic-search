from typing import List

import gradio as gr
import numpy as np
from pymilvus import MilvusClient

from semantic_search.utils import DB_NAME, model, search_params

CLIENT = MilvusClient(f"{DB_NAME}.db")
CLIENT.load_collection(DB_NAME)


def search_documents(query: str, top_k: int) -> str:
    """Search documents in the Milvus vector database regarding the query.
    Args:
        query (str): search query
        top_k (int): number of results to return
    Returns:
        str: formatted search results
    """
    embedded_query: List[np.ndarray] = model().encode_queries([query])

    results = CLIENT.search(
        collection_name=DB_NAME,
        data=embedded_query,
        search_params=search_params(),
        limit=top_k,
        output_fields=["id", "title", "snippet"],
    )

    # Format results
    formatted_results = []
    for hits in results:
        for hit in hits:
            result = {
                "id": hit.get("id"),
                "title": hit.get("entity", {}).get("title", ""),
                "snippet": hit.get("entity", {}).get("snippet", ""),
                "score": hit.get("distance"),
            }
            formatted_results.append(result)

    output = ""
    for idx, result in enumerate(formatted_results, 1):
        output += f"Result {idx}:\n"
        output += f"ID: {result['id']}\n"
        output += f"Title: {result['title']}\n"
        output += f"Snippet: {result['snippet']}\n"
        output += f"Score: {result['score']:.4f}\n"
        output += "-" * 50 + "\n"

    return output


if __name__ == "__main__":
    iface = gr.Interface(
        fn=search_documents,
        inputs=[
            gr.Textbox(label="Enter your search query"),
            gr.Slider(
                minimum=1, maximum=20, value=5, step=1, label="Number of results"
            ),
        ],
        outputs=gr.Textbox(label="Search Results", lines=10),
        title="Semantic Search Demo",
        description="Enter a query to search through the document collection using semantic similarity.",
    )

    iface.launch()
