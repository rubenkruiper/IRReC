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


class Retrieval(BaseSettings):
    ner_url: str = 'http://spar:8501/'
    classifier_url: str = 'http://classifier:8502/'
    top_k: int = 15


class SparseSettings(BaseSettings):
    type: str = 'bm25'


class FieldsAndWeights(BaseSettings):
    content: float = 1.0
    doc_title: float = 1.0
    NER_labels: float = 0.0
    filtered_NER_labels: float = 0.0
    filtered_NER_labels_domains: float = 1.0
    neighbours: float = 0.0
    bm25_weight: float = 1.0        # relative influence of BM25 results


class Indexing(BaseSettings):
    indexing_type: str = 'sparse'
    index_name: str = 'with_de'         # we lower case this, vaguely remember capitals caused issues with elasticsearch
    sparse_settings: SparseSettings
    fields_and_weights: FieldsAndWeights


class QueryExpansion(BaseSettings):
    haystack_url: str = 'http://haystack:8500/'
    prf_weight: float = 0.0
    kg_weight: float = 1.0
    nn_weight: float = 1.0


class Settings(BaseSettings):
    retrieval: Retrieval
    indexing: Indexing
    query_expansion: QueryExpansion
    recreate_sparse_index: bool = False
    recreate_dense_index: bool = False


class QueryExanderFromSettings:
    def __init__(self):
        self.haystack_url, self.ner_url, self.classifier_url = None, None, None
        self.indexing_type = "hybrid"
        self.fields_and_weights = {}
        # initialize which QE candidates/results to use
        self.prf_weight = 0         # relative influence of prf candidates
        self.kg_weight = 0          # relative influence of KG candidates
        self.nn_weight = 0          # relative influence of NN candidates

        self.was_initialised = False
        self.update_from_file()
        self.QE_obj = QueryExpander(self.classifier_url, self.ner_url,
                                    prf_weight=self.prf_weight,
                                    kg_weight=self.kg_weight,
                                    nn_weight=self.nn_weight)

    def update_from_file(self):
        with open("/data/information_retrieval_settings.json") as f:
            settings_from_file = json5.load(f)
        self.update_from_dict(settings_from_file)

    def update_from_dict(self, settings):
        self.haystack_url = settings["query_expansion"]["haystack_url"]
        self.ner_url = settings["retrieval"]["ner_url"]
        self.classifier_url = settings["retrieval"]["classifier_url"]
        if self.classifier_url not in ["", "no", "No", "None", "none"] and self.was_initialised:
            requests.post(f"{self.classifier_url}train/")
        else:
            # When initializing QE, the classifier isn't ready yet. Later we'd like to update the classifier if needed.
            self.was_initialised = True
        self.indexing_type = settings["indexing"]["sparse_settings"]["type"]

        self.fields_and_weights = settings["indexing"]["fields_to_index_and_weights"]
        self.prf_weight = settings["query_expansion"]["prf_weight"]
        self.kg_weight = settings["query_expansion"]["kg_weight"]
        self.nn_weight = settings["query_expansion"]["nn_weight"]

        self.QE_obj = QueryExpander(self.classifier_url, self.ner_url,
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
def update_weights(settings: Settings = None) -> dict:
    if not settings:
        QE_s.update_from_file()
    else:
        # condensed version of the settings that actually apply
        setting_dict = {
            'retrieval': {
                'ner_url': settings.retrieval.ner_url,
                'classifier_url': settings.retrieval.classifier_url,
                "top_k": settings.retrieval.top_k
            },
            'indexing': {
                'indexing_type':  settings.indexing.indexing_type,
                'index_name':  settings.indexing.index_name.lower(),
                'sparse_settings': {
                    'type': settings.indexing.sparse_settings.type
                },
                'fields_to_index_and_weights': {
                    'content': settings.indexing.fields_and_weights.content,
                    'doc_title': settings.indexing.fields_and_weights.doc_title,
                    'NER_labels': settings.indexing.fields_and_weights.NER_labels,
                    'filtered_NER_labels': settings.indexing.fields_and_weights.filtered_NER_labels,
                    'filtered_NER_labels_domains': settings.indexing.fields_and_weights.filtered_NER_labels_domains,
                    'neighbours': settings.indexing.fields_and_weights.neighbours,
                    'bm25_weight': settings.indexing.fields_and_weights.bm25_weight
                }
            },
            'query_expansion': {
                'haystack_url': settings.query_expansion.haystack_url,
                'prf_weight': settings.query_expansion.prf_weight,
                'kg_weight': settings.query_expansion.kg_weight,
                'nn_weight': settings.query_expansion.nn_weight
            }
        }
        QE_s.update_from_dict(setting_dict)

        retriever_setting_dict = {
            "indexing_type": settings.indexing.indexing_type,
            "index_name": settings.indexing.index_name,
            "sparse_type": settings.indexing.sparse_settings.type,
            "content": settings.indexing.fields_and_weights.content,
            "doc_title": settings.indexing.fields_and_weights.doc_title,
            "NER_labels": settings.indexing.fields_and_weights.NER_labels,
            "filtered_NER_labels": settings.indexing.fields_and_weights.filtered_NER_labels,
            "filtered_NER_labels_domains": settings.indexing.fields_and_weights.filtered_NER_labels_domains,
            "neighbours": settings.indexing.fields_and_weights.neighbours,
            "bm25_weight": settings.indexing.fields_and_weights.bm25_weight,
            "top_k": settings.retrieval.top_k,
            "recreate_sparse_index": settings.recreate_sparse_index,
            "recreate_dense_index": settings.recreate_dense_index,
            'ner_url': settings.retrieval.ner_url,
            'classifier_url': settings.retrieval.classifier_url
        }

        r = requests.post(f"{QE_s.haystack_url}set_field_weights/", json=retriever_setting_dict).json()
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
    response = requests.post(f"{QE_s.haystack_url}search/",
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
    if QE_s.ner_url in ["", "No", "no", "None", "none"]:
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
        print("[Running with PRF!]")
        print("[Running with PRF!]")
        print("[Running with PRF!]")
        print("[Running with PRF!]")
        combined_pred = regular_query(query)
        expanded_query, qe_insight = QE_s.QE_obj.expand_query(query, combined_pred)
    else:
        expanded_query, qe_insight = QE_s.QE_obj.expand_query(query)

    return {"query": query,
            "expanded_query": expanded_query,
            "qe_insight": qe_insight}