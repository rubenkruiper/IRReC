from typing import Optional, Dict, List, Union, Any
import json5, pickle

from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from pydantic import validator, BaseSettings, BaseModel, Required
# from torch.cuda import is_available, device_count
from transformers import BertModel, BertTokenizer

from embedder import Embedder
from classifier import Classifier


class Settings(BaseSettings):
    """
    The same as when declaring query parameters, when a model attribute has a default value, it is not required.
    Otherwise, it is required. Use None to make it just optional.
    """
    bert_model: Optional[str] = "whaleloops/phrase-bert"
    cache_dir: Optional[str] = "data/pretrained_embeddings_cache/"
    classifier_dir: Optional[str] = "data/classifier_data/"
    IDF_path: Optional[str] = None
    embedding_folder_name: Optional[str] = "default_embedding_settings"
    layers_to_use: List[int] = [12]
    layer_combination: str = "avg"
    idf_threshold: float = 0.1  # actually, if we omit all tokens due to the idf_threshold we run into an error
    idf_weight_factor: float = 1.0
    not_found_idf_value: float = 0.5
    num_neighbours: Optional[int] = 500
    tfidf_cutoff: Optional[float] = 0.6
    num_neighbours_cutoff: Optional[float] = 250
    foreground_corpus_terms: Optional[str] = "foreground_objects.pkl"
    background_corpus_terms: Optional[str] = "background_objects.pkl"
    top_k_semantically_similar: Optional[int] = 5
    metric: Optional[str] = "cosine"

    # Some basic additional validation
    @validator('cache_dir', 'cluster_dir') #, 'IDF_path') >> we now allow creating IDF paths from scratch
    def validate_paths(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Path doesn't exist, check again: {v}")
        return v

    @validator('num_clusters')
    def validate_num_clusters(cls, v):
        if not 1 < v < 30000:
            raise ValueError("In the api the max value for K is set to 30.000")
        return v

    # @validator('gpu')
    # def check_gpu_exists(cls, v):
    #     if is_available():
    #         devs = device_count()
    #         if not v <= devs:
    #             raise ValueError(f"Selected gpu ({v}) not available, number of Nvidia gpus: {devs}")
    #     return v


class ToPredict(BaseModel):
    spans: Union[List[str], str]


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

    def grab_settings(self):
        with open(self.path_to_settings) as f:
            all_settings = json5.load(f)
        keys_to_keep = ["bert_model", "cache_dir", "classifier_dir", "IDF_path"]
        selected_settings = {k:v for k, v in all_settings.items() if k in keys_to_keep}
        group_to_keep = ["embedding_settings", "classifier_settings"]
        selected_settings += {sub_k: sub_v for k, v in all_settings for sub_k, sub_v in v.items() if k in group_to_keep}
        return Settings(**selected_settings)

    def initialize_embeddings_from_settings(self):
        """ Initializes an Embedder model. """
        # Grab the latest version of settings/configuration
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


    def initialize_classifier_from_settings(self):
        """ Initializes a Classifier model, may need training. """
        # Grab the latest version of settings/configuration
        settings = self.grab_settings()

        # TODO; figure out how I want to do the classification
        self.classifier = Classifier(self.embedder,
                                     self.foreground_term_filepath,
                                     self.background_term_filepath,
                                     settings.top_k_semantically_similar
                                     )
        return settings


## Set up the Cluster_model and API
path_to_settings = "/data/information_retrieval_settings.json"
Cluster_api = FastAPI()
hub = EmbeddingHub()


@Cluster_api.get("/", status_code=200)
def root() -> dict:
    """
    The 'index' page shows the settings for the current cluster model
    """
    return hub.settings


@Cluster_api.post("/train/")
def train_classifier(classifier_settings: Dict[str, Any] = None):
    if not hub.embedder:
        hub.initialize_embeddings_from_settings()

    if not hub.classifier:
        hub.initialize_classifier_from_settings()

    classifier_nr_neighbours = classifier_settings['classifier_nr_neighbours'] if classifier_settings else 500
    tfidf_cutoff = classifier_settings['tfidf_cutoff'] if classifier_settings else .6
    num_neighbours_cutoff = classifier_settings['num_neighbours_cutoff'] if classifier_settings else 250
    hub.classifier.train_classifier_from_heuristics(n_neighbours=classifier_nr_neighbours,
                                                    min_tfidf_value=tfidf_cutoff,
                                                    min_num_foreground_neighbours=num_neighbours_cutoff)


@Cluster_api.post("/filter_non_domain_spans/")
def filter_non_domain_spans(to_be_predicted: ToPredict) -> Dict[str, str]:
    """
    Returns a `list` of terms from the assigned cluster.
    """
    if to_be_predicted.spans:
        domain_spans = hub.classifier.predict_domains(to_be_predicted.spans)
        return {'domain_spans': domain_spans}
    # If the input is None, simply return an empty list
    return {'domain_spans': []}


@Cluster_api.post("/get_neighbours/")
def get_neighbours(to_be_predicted: ToPredict) -> Dict[str, str]:
    """
    Returns a `list` of terms from the assigned cluster.
    """
    if to_be_predicted.spans:
        neighbours = hub.classifier.get_nearest_neighbours(to_be_predicted.spans)
        return {'neighbours': neighbours}
    # If the input is None, simply return an empty list
    return {'neighbours': []}


@Cluster_api.get("/get_idf_weights/{input_str}")
def get_idf_weights(input_str: Optional[str] = Query(default=Required, max_length=1000)):
    """
    Returns the query and it's corresponding idf_weights
    """
    if input_str:
        tokens, indices = hub.embedder.prepare_tokens(input_str)
        idf_weights = hub.embedder.get_idf_weights_for_indices(tokens, indices).tolist()
        return {"idf_weights": idf_weights}
    return {"idf_weights": []}


# Would need to pass the new settings as params={}, reading from file for now
@Cluster_api.post("/update_settings/")
def update_settings(new_settings: Settings) -> None:
    hub.initialize_classifier_from_settings(new_settings)
    return hub.settings

