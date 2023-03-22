import json5
import requests
import logging
from urllib import parse

from fastapi import FastAPI
from pydantic import BaseSettings, BaseModel
from fastapi.middleware.cors import CORSMiddleware

from query_expander import QueryExpander
from query_utils import *


# set requests and urllib3 logging to Warnings only todo; not sure if this helps if implemented here only
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class FieldsAndWeights(BaseSettings):
    content: float = 1.0
    doc_title: float = 1.0
    NER_labels: float = 0.0
    filtered_NER_labels: float = 0.0
    filtered_NER_labels_domains: float = 1.0
    neighbours: float = 0.0
    bm25_weight: float = 1.0        # relative influence of BM25 results
    top_k: int = 50


class QueryExpansionWeights(BaseSettings):
    prf_weight: float = 0.0
    kg_weight: float = 1.0
    nn_weight: float = 1.0


class Settings(BaseSettings):
    haystack_endpoint: str = 'http://haystack:8500/'
    ner_endpoint: str = 'http://spar:8501/'
    classifier_endpoint: str = 'http://classifier:8502/'
    indexing_type: str = "hybrid"
    sparse_type: str = "bm25f"
    recreate_sparse_index: bool = False
    recreate_dense_index: bool = False
    fields_and_weights: FieldsAndWeights
    query_expansion: QueryExpansionWeights


class QueryExanderFromSettings:
    def __init__(self):
        self.haystack_endpoint, self.ner_endpoint, self.classifier_endpoint = None, None, None
        self.indexing_type = "hybrid"
        self.recreate_sparse_index = False
        self.recreate_dense_index = False
        self.fields_and_weights = {}
        # initialize which QE candidates/results to use
        self.prf_weight = 0         # relative influence of prf candidates
        self.kg_weight = 0          # relative influence of KG candidates
        self.nn_weight = 0          # relative influence of NN candidates

        self.update_from_file()
        self.QE_obj = QueryExpander(self.classifier_endpoint, self.ner_endpoint,
                                    prf_weight=self.prf_weight,
                                    kg_weight=self.kg_weight,
                                    nn_weight=self.nn_weight)

    def update_from_file(self):
        with open("/data/information_retrieval_settings.json") as f:
            settings_from_file = json5.load(f)
        self.update_from_dict(settings_from_file)

    def update_from_dict(self, settings):
        self.haystack_endpoint = settings["query_expansion"]["haystack_endpoint"]
        self.ner_endpoint = settings["retrieval"]["ner_url"]
        self.classifier_endpoint = settings["retrieval"]["classifier_url"]
        self.indexing_type = settings["indexing"]["sparse_settings"]["type"]
        self.recreate_index = settings["indexing"]["sparse_settings"]["recreate_sparse_index"]
        self.recreate_index = settings["indexing"]["dense_settings"]["recreate_dense_index"]

        self.fields_and_weights = settings["indexing"]["fields_to_index_and_weights"]
        self.prf_weight = settings["query_expansion"]["prf_weight"]
        self.kg_weight = settings["query_expansion"]["kg_weight"]
        self.nn_weight = settings["query_expansion"]["nn_weight"]

        self.QE_obj = QueryExpander(self.classifier_endpoint, self.ner_endpoint,
                                    prf_weight=self.prf_weight,
                                    kg_weight=self.kg_weight,
                                    nn_weight=self.nn_weight)


# Set up the Query Expander object from settings file, as well as the API
QE_s = QueryExanderFromSettings()
QE_api = FastAPI()
#
# origins = [
#     "http://localhost:3000",
#     "https://localhost:3000",
#     "https://localhost:8000",
#     "https://localhost:8503",
# ]

QE_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@QE_api.post("/update_weights/")
def update_weights(pydantic_settings: Settings = None) -> dict:
    if not pydantic_settings:
        QE_s.update_from_file()
    else:
        # TODO; change this dict! so it makes senese... or just load from file?
        # TODO; change this dict! so it makes senese...
        # TODO; change this dict! so it makes senese...
        # TODO; change this dict! so it makes senese...
        settings = {
            'haystack_endpoint': pydantic_settings.haystack_endpoint,
            'ner_endpoint': pydantic_settings.ner_endpoint,
            'classifier_endpoint': pydantic_settings.classifier_endpoint,
            'indexing_type': pydantic_settings.indexing_type,
            'sparse_type': pydantic_settings.sparse_type,
            'recreate_sparse_index': pydantic_settings.recreate_sparse_index,
            'recreate_dense_index': pydantic_settings.recreate_dense_index,
            'query_expansion': {
                'prf_weight': pydantic_settings.query_expansion.prf_weight,
                'kg_weight': pydantic_settings.query_expansion.kg_weight,
                'nn_weight': pydantic_settings.query_expansion.nn_weight
            },
            'fields_to_index_and_weights': {
                'content': pydantic_settings.fields_and_weights.content,
                'doc_title': pydantic_settings.fields_and_weights.doc_title,
                'NER_labels': pydantic_settings.fields_and_weights.NER_labels,
                'filtered_NER_labels': pydantic_settings.fields_and_weights.filtered_NER_labels,
                'filtered_NER_labels_domains': pydantic_settings.fields_and_weights.filtered_NER_labels_domains,
                'neighbours': pydantic_settings.fields_and_weights.neighbours,
                'bm25_weight': pydantic_settings.query_expansion.bm25_weight,
                'top_k': pydantic_settings.fields_and_weights.top_k}
        }
        QE_s.update_from_dict(settings)

    r = requests.post(f"{QE_s.haystack_endpoint}set_field_weights/", json=QE_s.retrieval_settings).json()
    r["prf_weight"] = QE_s.prf_weight
    r["kg_weight"] = QE_s.kg_weight
    r["nn_weight"] = QE_s.nn_weight
    return r


def quick_duplicate_finder(combined_pred):
    """ Avoid returning duplicate documents """
    unique_ids = []
    unique_passages = []
    deduplicated_combined_pred = {}
    for doc_title in combined_pred:
        retrieved_doc = combined_pred[doc_title]
        duplicate_doc = False
        duplicate_passage = False
        # check if the identifier was already found before
        identifier = retrieved_doc.potential_id
        if identifier not in unique_ids:
            unique_ids.append(identifier)
        elif identifier:
            duplicate_doc = True

        # check if any of the returned passages was already found (exact match)
        for retrieved_passage in retrieved_doc.contents:
            if retrieved_passage.text not in unique_passages:
                unique_passages.append(retrieved_passage.text)
            else:
                duplicate_passage = True

        if not duplicate_doc and not duplicate_passage:
            deduplicated_combined_pred[doc_title] = combined_pred[doc_title]
    return deduplicated_combined_pred


def regular_query(query: str):
    if not query.strip():
        # empty query
        return []
    chars_to_remove = """'"[]{}()\\/"""
    for char in chars_to_remove:
        query = query.replace(char, '')
    response = requests.post(f"{QE_s.haystack_endpoint}search/",
                             json={"query": query.strip()}).json()
    pipeline_predictions, query_time = response["result"], response["query_time"]
    combined_pred = combine_results_from_various_indices(pipeline_predictions, QE_s.fields_and_weights)

    # remove duplicate results, if any
    combined_pred = quick_duplicate_finder(combined_pred)

    # sort results by sum of scores for each document
    ranked_tuples = []
    for doc_title in combined_pred.keys():
        score = combined_pred[doc_title].sum_of_scores
        ranked_tuples.append((score, combined_pred[doc_title].as_dict()))

    ranked_tuples = sorted(ranked_tuples, key=lambda x: x[0], reverse=True)
    ranked_scores, _ = zip(*ranked_tuples)
    return [ranked_tuples[idx] for idx, s in enumerate(ranked_scores) if s not in ranked_scores[:idx]]


@QE_api.post("/q/{query}")
def search(query: str) -> dict:
    """
    IR using regular query
    """
    combined_pred = regular_query(query)
    return {"query": query, "combined_prediction": combined_pred}


@QE_api.post("/qe/{query}")
def expanded_query_search(query: str) -> Dict:
    """
    IR using expanded query
    """
    if QE_s.ner_endpoint in ["", "No", "no", "None", "none"]:
        return {"query": "You do not have an NER endpoint set"}

    if QE_s.prf_weight:
        combined_pred = regular_query(query)
        expanded_query, qe_insight = QE_s.QE_obj.expand_query(query, combined_pred)
    else:
        expanded_query, qe_insight = QE_s.QE_obj.expand_query(query)

    expanded_pred = regular_query(expanded_query)
    return {"query": query,
            # "combined_prediction": combined_pred,
            "expanded_query": expanded_query,
            "qe_insight": qe_insight,
            "expanded_prediction": expanded_pred}


@QE_api.post("/just_expand/{query}")
def just_expand(query: str) -> Dict:
    """
    Only expand the query, to see what the output looks like
    """
    if QE_s.prf_weight:
        print("[Running with PRF]")
        print("[Running with PRF]")
        print("[Running with PRF]")
        print("[Running with PRF]")
        combined_pred = regular_query(query)
        expanded_query, qe_insight = QE_s.QE_obj.expand_query(query, combined_pred)
    else:
        expanded_query, qe_insight = QE_s.QE_obj.expand_query(query)

    return {"query": query,
            "expanded_query": expanded_query,
            "qe_insight": qe_insight}