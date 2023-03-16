import json5
import requests
from urllib import parse

from fastapi import FastAPI
from pydantic import BaseSettings
from fastapi.middleware.cors import CORSMiddleware

from query_expander import QueryExpander
from query_utils import *


class FieldsAndWeights(BaseSettings):
    indexing_type: str = 'hybrid'
    sparse_type: str = 'bm25f'
    content: float = 1.0
    doc_title: float = 1.0
    SPaR_labels: float = 0.0
    filtered_SPaR_labels: float = 0.0
    filtered_SPaR_labels_domain: float = 1.0
    neighbours: float = 0.0
    bm25: float = 1.0
    top_k: int = 50


class QueryExpansionWeights(BaseSettings):
    prf_weight: float = 0.0
    kg_broader_weight: float = 0.0
    kg_narrower_weight: float = 0.0
    kg_alternatives_weight: float = 0.0
    nn_weight: float = 1.0


class Settings(BaseSettings):
    sparql_endpoint: str = 'http://host.docker.internal:7200/repositories/BRE'
    haystack_endpoint: str = 'http://haystack:8500/'
    spar_endpoint: str = 'http://spar:8501/'
    cluster_endpoint: str = 'http://cluster:8502/'
    retrieval_settings: FieldsAndWeights
    qe_candidates: QueryExpansionWeights



class QueryExanderFromSettings:
    def __init__(self):
        self.sparql_endpoint, self.haystack_endpoint, self.spar_endpoint, self.cluster_endpoint = None, None, None, None
        self.retrieval_settings = {}
        self.indexing_type = "hybrid"
        # initialize which QE candidates to use
        self.prf_weight = 0
        self.kg_broader_weight = 0
        self.kg_narrower_weight = 0
        self.kg_alternatives_weight = 0
        self.nn_weight = 0

        self.update_from_file()
        self.QE_obj = QueryExpander(self.sparql_endpoint, self.cluster_endpoint, self.spar_endpoint,
                                    self.prf_weight, self.kg_narrower_weight, self.kg_alternatives_weight,
                                    self.nn_weight)

    def update_from_file(self):
        with open("/data/query_expansion_settings.json") as f:
            settings_from_file = json5.load(f)
        self.update_from_dict(settings_from_file)

    def update_from_dict(self, settings):
        self.sparql_endpoint = settings["sparql_endpoint"]
        self.haystack_endpoint = settings["haystack_endpoint"]
        self.spar_endpoint = settings["spar_endpoint"]
        self.cluster_endpoint = settings["cluster_endpoint"]
        self.retrieval_settings = settings["retrieval_settings"]

        self.prf_weight = settings["qe_candidates"]["prf_weight"]
        self.kg_broader_weight = settings["qe_candidates"]["kg_broader_weight"]
        self.kg_narrower_weight = settings["qe_candidates"]["kg_narrower_weight"]
        self.kg_alternatives_weight = settings["qe_candidates"]["kg_alternatives_weight"]
        self.nn_weight = settings["qe_candidates"]["nn_weight"]

        self.QE_obj = QueryExpander(self.sparql_endpoint, self.cluster_endpoint, self.spar_endpoint,
                                    prf_weight=self.prf_weight,
                                    kg_broader_weight=self.kg_broader_weight,
                                    kg_narrower_weight=self.kg_narrower_weight,
                                    kg_alternatives_weight=self.kg_alternatives_weight, nn_weight=self.nn_weight)


# Set up the Cluster_model and API
QE_s = QueryExanderFromSettings()
# QE_api = FastAPI(root_path="/brretrieval")
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
        settings = {
         'sparql_endpoint': pydantic_settings.sparql_endpoint,
         'haystack_endpoint': pydantic_settings.haystack_endpoint,
         'spar_endpoint': pydantic_settings.spar_endpoint,
         'cluster_endpoint': pydantic_settings.cluster_endpoint,
         'qe_candidates': {'prf_weight': pydantic_settings.qe_candidates.prf_weight,
                           'kg_broader_weight': pydantic_settings.qe_candidates.kg_broader_weight,
                           'kg_narrower_weight': pydantic_settings.qe_candidates.kg_narrower_weight,
                           'kg_alternatives_weight': pydantic_settings.qe_candidates.kg_alternatives_weight,
                           'nn_weight': pydantic_settings.qe_candidates.nn_weight},
         'retrieval_settings': {'indexing_type': pydantic_settings.retrieval_settings.indexing_type,
                                'sparse_type': pydantic_settings.retrieval_settings.sparse_type,
                                'content': pydantic_settings.retrieval_settings.content,
                                'doc_title': pydantic_settings.retrieval_settings.doc_title,
                                'SPaR_labels': pydantic_settings.retrieval_settings.SPaR_labels,
                                'filtered_SPaR_labels': pydantic_settings.retrieval_settings.filtered_SPaR_labels,
                                'cluster_filtered': pydantic_settings.retrieval_settings.cluster_filtered,
                                'cluster_neighbours': pydantic_settings.retrieval_settings.cluster_neighbours,
                                'bm25': pydantic_settings.retrieval_settings.bm25,
                                'top_k': pydantic_settings.retrieval_settings.top_k}}
        QE_s.update_from_dict(settings)

    r = requests.post(f"{QE_s.haystack_endpoint}set_field_weights/", json=QE_s.retrieval_settings).json()
    r["prf_weight"] = QE_s.prf_weight
    r["kg_broader_weight"] = QE_s.kg_broader_weight
    r["kg_narrower_weight"] = QE_s.kg_narrower_weight
    r["kg_alternatives_weight"] = QE_s.kg_alternatives_weight
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


def regular_query(query):
    if not query.strip():
        # empty query
        return []
    query = query.replace('"', '').replace("'", '').replace('(', '').replace(')', '').replace('/', '').replace('\\', '')
    url_query = parse.quote(query)
    response = requests.get(f"{QE_s.haystack_endpoint}search/{url_query}").json()
    pipeline_predictions, query_time = response["result"], response["query_time"]
    combined_pred = combine_results_from_various_indices(pipeline_predictions, QE_s.retrieval_settings)

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