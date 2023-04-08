from typing import Optional, Dict, List, Union, Any
import json5, pickle, logging

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import validator, BaseSettings, BaseModel, Required
# from torch.cuda import is_available, device_count
from transformers import BertModel, BertTokenizer

from utils import *
from embedder import Embedder
from classifier import Classifier


# set requests and urllib3 logging to Warnings only todo; not sure if this helps if implemented here only
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class Settings(BaseSettings):
    """
    The same as when declaring query parameters, when a model attribute has a default value, it is not required.
    Otherwise, it is required. Use None to make it just optional.
    """
    bert_model: Optional[str] = "whaleloops/phrase-bert"
    cache_dir: Optional[str] = "/data/pretrained_embeddings_cache/"
    classifier_dir: Optional[str] = "/data/classifier_data/"
    IDF_path: Optional[str] = None
    embedding_folder_name: Optional[str] = "standard_settings"
    layers_to_use: List[int] = [12]
    layer_combination: str = "avg"
    idf_threshold: float = 0.1  # actually, if we omit all tokens due to the idf_threshold we run into an error
    idf_weight_factor: float = 1.0
    not_found_idf_value: float = 0.5
    classifier_nr_neighbours: Optional[int] = 500
    tfidf_cutoff: Optional[float] = 0.6
    num_neighbours_cutoff: Optional[float] = 375
    foreground_corpus_terms: Optional[str] = "foreground_terms.pkl"
    background_corpus_terms: Optional[str] = "background_terms.pkl"
    top_k_semantically_similar: Optional[int] = 5
    metric: Optional[str] = "cosine"

    # Some basic additional validation
    @validator('cache_dir', 'classifier_dir') #, 'IDF_path') >> we now allow creating IDF paths from scratch
    def validate_paths(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Path doesn't exist, check again: {v}")
        return v

    # @validator('gpu')
    # def check_gpu_exists(cls, v):
    #     if is_available():
    #         devs = device_count()
    #         if not v <= devs:
    #             raise ValueError(f"Selected gpu ({v}) not available, number of Nvidia gpus: {devs}")
    #     return v


class ToPredict(BaseModel):
    span_lists: List[List[str]] = None
    spans: List[str] = None


class EmbeddingHub:
    def __init__(self, path_to_settings: str = "/data/information_retrieval_settings.json"):
        """
        Object that provides access to functionalites like computing term embeddings, a kNN graph, top_k nearest
        neighbours, setting up and calling a domain classifier. Note that this object is initialised as a placeholder,
        after which the classifier can/will be initialised using self.intialize_model_from_settings().
        """
        self.tokenizer = None
        self.bert_model = None
        self.embedder = None
        self.classifier = None
        self.classifier_dir = None
        self.foreground_term_filepath = None
        self.background_term_filepath = None
        self.embeddings_dir = None
        self.path_to_settings = path_to_settings

        self.current_embeddings = None

    def grab_settings(self):
        with open(self.path_to_settings) as f:
            s = json5.load(f)
        keys_to_keep = ["bert_model", "cache_dir", "classifier_dir", "IDF_path"]
        selected_settings = {k: v for k, v in s.items() if k in keys_to_keep}
        group_to_keep = ["embedding_settings", "classifier_settings"]
        selected_settings.update({sub_k: sub_v for k in s if k in group_to_keep for sub_k, sub_v in s[k].items()})
        return Settings(**selected_settings)

    def initialize_embeddings_from_settings(self, settings: Optional[Settings] = None):
        """ Initializes an Embedder model. """
        # Grab the latest version of settings/configuration
        if not settings:
            settings = self.grab_settings()
        self.classifier_dir = settings.classifier_dir
        self.foreground_term_filepath = Path(self.classifier_dir).joinpath(settings.foreground_corpus_terms)
        self.background_term_filepath = Path(self.classifier_dir).joinpath(settings.background_corpus_terms)
        # Init Tokenizer, BertModel and Embedder
        self.tokenizer = BertTokenizer.from_pretrained(settings.bert_model)
        self.bert_model = BertModel.from_pretrained(settings.bert_model,
                                                    output_hidden_states=True,
                                                    cache_dir=settings.cache_dir)
        self.embeddings_dir = Path(settings.classifier_dir).joinpath(settings.embedding_folder_name)
        self.embeddings_dir.mkdir(exist_ok=True)
        self.embedder = Embedder(self.tokenizer, self.bert_model,
                                 IDF_dict=json5.load(open(settings.IDF_path)),
                                 embedding_dir=self.embeddings_dir,
                                 layers_to_use=settings.layers_to_use,
                                 layer_combination=settings.layer_combination,
                                 idf_threshold=settings.idf_threshold,
                                 idf_weight_factor=settings.idf_weight_factor,
                                 not_found_idf_value=settings.not_found_idf_value)

    def embed_list_of_terms(self,
                            max_num_cpu_threads: int = 4,
                            subset_size: int = 1000,
                            prefix: str = None,
                            list_of_terms: List[str] = None):
        """
        Call the initialized embedder. This is split into subsets so we don't overload memory (adjust values if needed).


        """
        if not list_of_terms:
            # if no list of terms is provided, we embed all of the terms from the fore and background corpus
            foreground_terms = pickle.load(open(self.foreground_term_filepath, 'rb'))
            background_terms = pickle.load(open(self.background_term_filepath, 'rb'))
            list_of_terms = list(set(foreground_terms + background_terms))
            self.embedder.embed_fore_and_background_terms(
                max_num_cpu_threads, subset_size, list_of_terms
            )
        else:
            # else, we expect that the fore and background embeddings exist
            self.embedder.embed_large_number_of_new_terms(
                max_num_cpu_threads, subset_size, prefix, list_of_terms
            )

    def initialize_classifier_from_settings(self, settings: Optional[Settings] = None):
        """ Initializes a Classifier model, may need training. """
        # Grab the latest version of settings/configuration
        if not settings:
            settings = self.grab_settings()
        self.initialize_embeddings_from_settings()
        self.classifier = Classifier(
            self.embedder,
            self.foreground_term_filepath,
            self.background_term_filepath,
            settings.top_k_semantically_similar
        )
        return settings


# Set up the Cluster_model and API
path_to_settings = "/data/information_retrieval_settings.json"
Classifier_api = FastAPI()
hub = EmbeddingHub()


@Classifier_api.get("/", status_code=200)
def root() -> dict:
    """
    The 'index' page shows the settings for the current cluster model
    """
    return hub.grab_settings()


@Classifier_api.post("/train/")
def train_classifier(classifier_settings: Dict[str, Any] = None):
    if not hub.embedder:
        hub.initialize_embeddings_from_settings()
        hub.embed_list_of_terms()

    if not hub.classifier:
        hub.initialize_classifier_from_settings()

    classifier_nr_neighbours = classifier_settings['classifier_nr_neighbours'] if classifier_settings else 500
    tfidf_cutoff = classifier_settings['tfidf_cutoff'] if classifier_settings else .6
    num_neighbours_cutoff = classifier_settings['num_neighbours_cutoff'] if classifier_settings else 250
    hub.classifier.train_classifier_from_heuristics(classifier_nr_neighbours=classifier_nr_neighbours,
                                                    min_tfidf_value=tfidf_cutoff,
                                                    min_num_foreground_neighbours=num_neighbours_cutoff)


@Classifier_api.post("/filter_non_domain_spans/")
def filter_non_domain_spans(to_be_predicted: ToPredict):
    """
    Returns a `list` of terms from the assigned cluster.
    """
    if to_be_predicted.spans:
        domain_spans = hub.classifier.predict_domains(to_be_predicted.spans)
        return {'domain_spans': domain_spans}
    elif to_be_predicted.span_lists:    # todo; put this in a separte helper function
        lists_to_return = predict_uniques_all_contents(to_be_predicted.span_lists,
                                                       hub.classifier.predict_domains)
        return {'domain_spans': lists_to_return}
    # If the input is None, simply return an empty list
    return {'domain_spans': []}


@Classifier_api.post("/get_neighbours/")
def get_neighbours(to_be_predicted: ToPredict):
    """
    Returns a `list` of terms from the assigned cluster.
    """
    if to_be_predicted.spans:
        neighbours = hub.classifier.get_nearest_neighbours(to_be_predicted.spans)
        return {'neighbours': neighbours}
    elif to_be_predicted.span_lists:
        lists_to_return = predict_uniques_all_contents(to_be_predicted.span_lists,
                                                       hub.classifier.get_nearest_neighbours)
        return {'neighbours': lists_to_return}
    # If the input is None, simply return an empty list
    return {'neighbours': []}


@Classifier_api.post("/get_idf_weights/")
def get_idf_weights(to_be_predicted: ToPredict):
    """
    Returns the query and it's corresponding idf_weights
    """
    idf_weights = []
    if to_be_predicted.spans:
        for span in to_be_predicted.spans:
            tokens, indices = hub.embedder.prepare_tokens(span)
            idf_weights.append(hub.embedder.get_idf_weights_for_indices(tokens, indices).tolist())
    elif to_be_predicted.span_lists:
        for span_list in to_be_predicted.span_lists:
            idf_weights_sub_list = []
            for span in span_list:
                tokens, indices = hub.embedder.prepare_tokens(span)
                idf_weights_sub_list.append(hub.embedder.get_idf_weights_for_indices(tokens, indices).tolist())
            idf_weights.append(idf_weights_sub_list)

    return {"idf_weights": idf_weights}


# Would need to pass the new settings as params={}, reading from file for now
@Classifier_api.post("/update_settings/")
def update_settings(new_settings: Settings) -> None:
    hub.initialize_classifier_from_settings(new_settings)
    return hub.settings

