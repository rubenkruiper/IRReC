import json5, time, logging
from typing import Optional
from fastapi import FastAPI, Query
from pydantic import Required, BaseModel

from information_retrieval_hub import InformationRetrievalHub

# set requests and urllib3 logging to Warnings only todo; not sure if this helps if implemented here only
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class PlainQuery(BaseModel):
    query: str


class RetrievalSettings(BaseModel):
    indexing_type: str = "hybrid"
    index_name: str = "with_de"
    sparse_type: str = "bm25f"
    content: float = 2.0
    doc_title: float = 1.0
    NER_labels: float = 1.0
    filtered_NER_labels: float = 1.0
    filtered_NER_labels_domains: float = 1.0
    neighbours: float = 1.0
    bm25_weight: float = 1.0
    top_k: int = 10
    recreate_sparse_index: bool = False
    recreate_dense_index: bool = False
    ner_url: str = 'http://spar:8501/'
    classifier_url: str = 'http://classifier:8502/'


# Set up the Pipeline
with open("/data/information_retrieval_settings.json") as f:
    initial_settings = json5.load(f)

my_pipeline = InformationRetrievalHub(initial_settings)
my_pipeline.set_up_pipelines()

## Set up the API
Retrieval_api = FastAPI()


@Retrieval_api.post("/search/")
def search(query: PlainQuery):
    start_time = time.time()
    result = my_pipeline.run_query(query.query)
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
        "bm25": retrieval_settings.bm25_weight,
        "top_k": retrieval_settings.top_k
    }
    my_pipeline.fields_and_weights = new_fields_and_weights
    my_pipeline.indexing_type = retrieval_settings.indexing_type
    my_pipeline.index_name = retrieval_settings.index_name.lower()
    my_pipeline.top_k_per_retriever = retrieval_settings.top_k
    my_pipeline.recreate_sparse_index = retrieval_settings.recreate_sparse_index
    my_pipeline.recreate_dense_index = retrieval_settings.recreate_dense_index
    my_pipeline.ner_url = retrieval_settings.ner_url
    my_pipeline.classifier_url = retrieval_settings.classifier_url

    # re-init the sparse pipeline and retriever:
    if retrieval_settings.indexing_type in ['sparse', 'hybrid']:
        my_pipeline.sparse_type = retrieval_settings.sparse_type
        my_pipeline.update_sparse_pipeline()

    return {"indexing_type": my_pipeline.indexing_type,
            "sparse_type": my_pipeline.sparse_type,
            "fields_and_weights": my_pipeline.fields_and_weights,
            "top_k_per_retriever": my_pipeline.top_k_per_retriever}


@Retrieval_api.post("/compute_idf/")
def search():
    my_pipeline.idf_computer.compute_or_load_idf_weights([my_pipeline.foreground_output_dir,
                                                          my_pipeline.background_output_dir],
                                                         overwrite=False)
