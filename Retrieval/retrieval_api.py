import json5, time
from typing import Optional
from fastapi import FastAPI, Query
from pydantic import Required, BaseModel

from information_retrieval_hub import InformationRetrievalHub


class RetrievalSettings(BaseModel):
    indexing_type: str = "hybrid"
    sparse_type: str = "bm25f"
    content: float = 2.0
    doc_title: float = 1.0
    NER_labels: float = 1.0
    filtered_NER_labels: float = 1.0
    filtered_NER_labels_domains: float = 1.0
    neighbours: float = 1.0
    bm25: float = 1.0
    top_k: int = 10


# Set up the Pipeline
with open("/data/information_retrieval_settings.json") as f:
    initial_settings = json5.load(f)

my_pipeline = InformationRetrievalHub(initial_settings)
my_pipeline.set_up_pipelines()

## Set up the API
Retrieval_api = FastAPI()


@Retrieval_api.get("/search/{input_str}")
def search(input_str: Optional[str] = Query(default=Required, max_length=1000)):
    start_time = time.time()
    result = my_pipeline.run_query(input_str)
    query_time = time.time() - start_time
    return {"result": result, "query_time": query_time}


@Retrieval_api.post("/set_field_weights/")
def set_field_weights(retrieval_settings: RetrievalSettings):
    """
    Takes new settings from the api and re-initializes the sparse Haystack Pipeline with these settings.
    """
    new_fields_and_weights = {
        "content": retrieval_settings.content,
        "doc_title": retrieval_settings.doc_title,
        "NER_labels": retrieval_settings.NER_labels,
        "filtered_NER_labels": retrieval_settings.filtered_NER_labels,
        "filtered_NER_labels_domains": retrieval_settings.filtered_NER_labels_domains,
        "neighbours": retrieval_settings.neighbours,
        "bm25": retrieval_settings.bm25,
        "top_k": retrieval_settings.top_k
    }
    my_pipeline.fields_and_weights = new_fields_and_weights
    my_pipeline.indexing_type = retrieval_settings.indexing_type
    my_pipeline.top_k_per_retriever = retrieval_settings.top_k

    # re-init the sparse pipeline and retriever:
    if retrieval_settings.indexing_type in ['sparse', 'hybrid']:
        my_pipeline.sparse_type = retrieval_settings.sparse_type
        my_pipeline.update_sparse_pipeline()

    return {"indexing_type": my_pipeline.indexing_type,
            "sparse_type": my_pipeline.sparse_type,
            "fields_and_weights": my_pipeline.fields_and_weights,
            "top_k_per_retriever": my_pipeline.top_k_per_retriever}
