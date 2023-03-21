import glob
import json
import os
import logging
import itertools
import pickle
import time
import requests
import subprocess
from tqdm import tqdm
from typing import List, Dict
from pathlib import Path

# my utilities
from utils.IDF_computer import IdfComputer
from utils.customdocument import CustomDocument
from utils.conversion_utils import convert_inputs
from utils.preprocessor_spartxt import SparPreProcessor

# Haystack imports
from haystack.document_stores import ElasticsearchDocumentStore, FAISSDocumentStore
from haystack.nodes import ElasticsearchRetriever, DensePassageRetriever
from haystack.pipelines import DocumentSearchPipeline

# Set up a logger to keep track of where/when the system crashes
logger = logging.getLogger("IRReC")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class InformationRetrievalHub:
    def __init__(self, configs: Dict[str, str]):
        """
        The InformationRetrievalHub initializes a Haystack pipeline with several custom components. The Hub reads
        its settings from a configuration file passed through `configs`, this allows it to be re-initialized with
        different settings more easily.

        :param configs: Dictionary holding the settings for each of the pipeline components, which are responsible for
                        preprocessing inputs and indexing them. Currently also holds the default settings for query
                        expansion.
        """
        # keep track of original configuration (default settings for querying)
        self.configs = configs
        self.cache_dir = configs["cache_dir"]
        self.top_k_per_retriever = configs['retrieval']['top_k']

        # initialise the input/output paths and names for models to use
        self.bert_model_name = configs['bert_model']
        self.IDF_path = Path(configs["IDF_path"])
        self.idf_computer = IdfComputer(Path(configs["IDF_path"]))
        self.classifier_dir = Path(configs["classifier_dir"])
        self.foreground_terms_filename = configs["classifier_settings"]["foreground_corpus_terms"]
        self.background_terms_filename = configs["classifier_settings"]["background_corpus_terms"]
        self.embedding_dir = Path(configs["embedding_settings"]['embedding_folder_name'])

        # urls for SPaR and Classifier APIs
        self.ner_url = configs["retrieval"]["ner_url"]
        self.classifier_url = configs["retrieval"]["classifier_url"]

        # SPaR.txt preprocessor
        self.preprocessor = SparPreProcessor(self.ner_url, configs['filtering']['regex_filter'])

        # check GPU availability
        try:
            subprocess.check_output('nvidia-smi')
            self.use_gpu = True
        except:
            self.use_gpu = False

        # conversion type, input and output directory for foreground and background corpus
        self.foreground_conversion_type = configs["foreground_conversion_type"]
        self.foreground_input_dir = Path(configs["foreground_corpus_dir"])
        self.background_conversion_type = configs["background_conversion_type"]
        self.background_input_dir = Path(configs["background_corpus_dir"])
        if self.foreground_conversion_type not in ["pdf", "html"] or \
           self.background_conversion_type not in ["pdf", "html"]:
            raise Exception("choose conversion that starts with 'pdf' or 'html' (need to update 'xml')")
        self.foreground_output_dir = self.foreground_input_dir.parent.joinpath(self.foreground_input_dir.stem +
                                                                               "_converted")
        self.background_output_dir = self.background_input_dir.parent.joinpath(self.background_input_dir.stem +
                                                                               "_converted")

        # determine the type of indexing, which fields to index, and the name of the index
        self.indexing_type = configs["indexing"]["indexing_type"]  # sparse, hybrid or dense
        self.fields_and_weights = configs["indexing"]["fields_to_index_and_weights"]
        self.recreate_index = configs["indexing"]["recreate_index"]
        if self.indexing_type in ['hybrid', 'sparse']:
            self.sparse_type = configs["indexing"]["sparse_settings"]["type"]  # multi_match or plain
            self.sparse_index_name = configs["indexing"]["sparse_settings"]["index_name"]
            self.recreate_index = configs["indexing"]["sparse_settings"]["recreate_index"]

        # whether to apply weights to the fields indexed with FAISS
        self.dense_field_weights = configs["indexing"]["dense_settings"]["weigh_fields"]
        if self.indexing_type in ['hybrid', 'dense']:
            # dense config: FAISS standard configs from Retrieval
            self.hidden_dims = configs["indexing"]["dense_settings"]["hidden_dims"]
            self.faiss_index_type = configs["indexing"]["dense_settings"]["faiss_index"]
            self.faiss_model_format = configs["indexing"]["dense_settings"]["model_format"]
            self.max_seq_len_query = configs["indexing"]["dense_settings"]["max_seq_len_query"]
            self.max_seq_len_passage = configs["indexing"]["dense_settings"]["max_seq_len_passage"]
            self.batch_size = configs["indexing"]["dense_settings"]["batch_size"]

        # container to keep track of our pipelines
        self.pipelines = {}
        self.pipelines_to_query = []

    def set_up_pipelines(self):
        """
        Helper function to call the setup functions for each pipeline that has been specified in `configs.indexing`.
        """
        if self.indexing_type in ['sparse', 'hybrid']:
            self.pipelines['bm25'], self.pipelines['bm25f'] = self.set_up_sparse_pipelines(self.recreate_index)

        if self.indexing_type in ['dense', 'hybrid']:
            for field_to_index in self.fields_and_weights.keys():
                # set up a pipeline for each dense index
                self.pipelines[field_to_index] = self.set_up_dense_pipeline(field_to_index)

        # make sure next iteration the system checks that all SPaR labels are computed, before we start clustering
        with open("/data/spar_process_state.txt", 'w') as f:
            f.write("running")

        self.pipelines_to_query = self.pipelines.keys()

    def prepare_pipeline_inputs(self):
        """
        This method calls the preprocessing functionality (splitting, cleaning, SPaR.txt labelling, NNs, domain class).
        """
        # (1) Convert foreground corpus and background corpus
        convert_inputs(self.foreground_input_dir, self.foreground_output_dir, self.foreground_conversion_type)
        convert_inputs(self.background_input_dir, self.background_output_dir, self.background_conversion_type)

        # (2) Compute IDF weights over the converted corpora
        idf_values = self.idf_computer.compute_or_load_idf_weights([self.foreground_output_dir,
                                                                    self.background_output_dir],
                                                                   overwrite=False)

        # (3) Preprocess (run SPaR.txt, filter extracted terms, split everything into index-able chunks)
        #  - can pass custom split_length here, but avoiding this for now
        self.preprocess_inputs(self.foreground_output_dir, self.classifier_dir.joinpath(self.foreground_terms_filename))
        self.preprocess_inputs(self.background_output_dir, self.classifier_dir.joinpath(self.background_terms_filename))

        # (4) SPaR.txt is done, we'll do some domain classification and find NNs
        # todo; do I want to pass specific settings here? json= {'num_neighbours':x, etc...}
        requests.post(f"{self.classifier_url}train/")
        self.expand_documents(self.foreground_output_dir)

    def preprocess_inputs(self,
                          converted_documents_dir: Path,
                          term_output_path: Path,
                          split_length: int = 100) -> List[CustomDocument]:
        """
        Splits and labels the text found in pdfs during the conversion step. Splits long texts into passages and
        sorts out which NER labels occur in the splits for indexing.

        :param converted_documents_dir: Directory where `CustomDocument` have been written to during conversion.
        :param term_output_path:    The outputpath to store all the foreground or background terms.
        :param split_length:    Maximum length of text-splits, which is passed to the preprocessor. Haystack recommends
                                setting this to 500 for sparse indexing, and 100 for dense indexing. Thus hybrid would
                                require 100 as well.
        """
        # recursively look inside subfolders if they exist
        converted_document_filepaths = [filepath for filepath in converted_documents_dir.glob("*.json")]

        # (1) identify NER labels
        if self.ner_url not in ["no", "No", "None", "none", ""]:
            all_NER_labels = []
            logger.info("[Preprocessor] Running SPaR.txt to collect potential NER labels.")
            for converted_document_filepath in tqdm(converted_document_filepaths):
                converted_document = CustomDocument.load_document(converted_document_filepath)
                if not converted_document:
                    logger.info(f"Issue loading converted document: {converted_document_filepath}")
                    continue

                if any([len(c.NER_labels) > 1 for c in converted_document.all_contents]):
                    logger.info(f"[Preprocessor] skipping, NER outputs already found in: {converted_document_filepath}")
                else:
                    logger.info(f"[Preprocessor] Identifying NER labels for: {converted_document_filepath}")
                    processed_list_of_dicts = self.preprocessor.process(converted_document.to_list_of_dicts())
                    converted_document.replace_contents(processed_list_of_dicts)
                    converted_document.write_document()
                # keep track of all the NER labels
                labels_in_document = [label for c in converted_document.all_contents for label in c.NER_labels]
                # if not labels_in_document:
                # todo we should probably delete files without labels, retaining for now
                all_NER_labels += labels_in_document

            # group all the (cleaned) terms extracted for a corpus and store for classification later
            cleaned_terms = self.preprocessor.cleaning_helper(all_NER_labels)
            with open(term_output_path, 'wb') as f:
                pickle.dump(cleaned_terms, f)
            filtered_labels = set(cleaned_terms)
        else:
            filtered_labels = set(pickle.load(open(term_output_path, 'rb')))

        # (2) identify which NER labels remain after cleaning
        logger.info("[Preprocessor] Retaining the filtered/cleaned SPaR.txt labels separately.")
        for filepath in tqdm(converted_document_filepaths):
            processed_document = CustomDocument.load_document(filepath)
            if any([len(c.filtered_NER_labels) > 1 for c in processed_document.all_contents]):
                logger.info("[Filtering] Skipping filtering, since filtered labels found in: {}".format(
                    processed_document.output_fp
                ))
            else:
                logger.info("[Filtering] Running regex and IDF-based filters on SPaR labels.")
                # Filtering based on regex and IDF
                for idx, content in enumerate(processed_document.all_contents):
                    if content.NER_labels:
                        # update the document in place
                        only_keep_filtered = [l for l in content.NER_labels if l in filtered_labels]
                        processed_document.all_contents[idx].set_filtered_ner_labels(only_keep_filtered)
                # update stored file
                processed_document.write_document()

    def expand_documents(self, converted_documents_dir: Path):
        """
        We want to keep track of the filtered and classified terms in a CustomDocument's contents. This simply updates
        the converted document at it's file location.

        :param converted_documents_dir:    The path where we can find out converted documents.
        :param term_output_path:    The path where all the filtered terms for this corpus are stored.
        """
        # grab the converted documents (at this stage expecting pre-processed)
        converted_document_filepaths = [filepath for filepath in converted_documents_dir.glob("*.json")]

        if self.classifier_url not in ["no", "No", "None", "none", ""]:
            for converted_document_filepath in converted_document_filepaths:
                converted_document = CustomDocument.load_document(converted_document_filepath)
                # make sure neighbours do not exist anywhere yet
                if any([len(c.neighbours) > 1 for c in converted_document.all_contents]):
                    logger.info(
                        f"[Classifier] skipping, classes and NNs already found in: {converted_document_filepath}")
                    continue
                else:
                    new_contents = []
                    for content in converted_document.all_contents:
                        if content.filtered_NER_labels:
                            domain_spans = requests.post(f"{self.classifier_url}filter_non_domain_spans/",
                                                         json={"spans": content.filtered_NER_labels}).json()
                            content.set_filtered_ner_label_domains(domain_spans["domain_spans"])
                            neighbours = requests.post(f"{self.classifier_url}get_neighbours/",
                                                       json={"spans": content.filtered_NER_labels}).json()
                            content.set_neighbours(neighbours["neighbours"])
                        new_contents.append(content)

                    # update the CustomDocument and save changes to file
                    converted_document.replace_contents(new_contents)
                    converted_document.write_document()

    def initialize_sparse_docstore(self, index_name: str):
        fields = ["content", "doc_title", "SPaR_labels", "filtered_SPaR_labels",
                  "filtered_SPaR_labels_domain", "neighbours"]

        # 'skip', 'overwrite' or 'fail'
        duplicate_documents = 'overwrite' if self.recreate_index else 'skip'

        # Start Elasticsearch using Docker via the Haystack utility function
        sparse_document_store = ElasticsearchDocumentStore(host="host.docker.internal",
                                                           index=index_name,
                                                           search_fields=fields,
                                                           content_field="content",
                                                           recreate_index=self.recreate_index,
                                                           duplicate_documents=duplicate_documents)
        return sparse_document_store

    def initialize_sparse_retriever(self, sparse_type, sparse_document_store: ElasticsearchDocumentStore):
        if sparse_type == "bm25f":           # "combined_fields"]: dropped for now
            fields = [f"content^{self.fields_and_weights['content']}",
                      f"doc_title^{self.fields_and_weights['doc_title']}",
                      f"SPaR_labels^{self.fields_and_weights['SPaR_labels']}",
                      f"filtered_SPaR_labels^{self.fields_and_weights['filtered_SPaR_labels']}",
                      f"filtered_SPaR_labels_domain^{self.fields_and_weights['filtered_SPaR_labels_domain']}",
                      f"neighbours^{self.fields_and_weights['neighbours']}"]

            custom_query = {
                "query": {
                    "multi_match": {
                        "query": "${query}",
                        "fields": fields,
                        "type": "cross_fields",
                    }
                }
            }

            custom_query_str = json.dumps(custom_query).replace('"${query}"', '${query}')
            sparse_retriever = ElasticsearchRetriever(document_store=sparse_document_store,
                                                      custom_query=custom_query_str)
        else:
            sparse_retriever = ElasticsearchRetriever(document_store=sparse_document_store)

        return sparse_retriever

    def update_sparse_pipeline(self):
        """
        Reinitialize the sparse pipeline, which means setting up the ES Document Store and the ES retriever again.
        We'll be using Haystacks base pipeline load_from_config for this.
        """
        # todo; assuming we always need/want to recreate the sparse index for now
        sparse_type = "bm25f" if self.sparse_type == "bm25f" else "bm25"
        sparse_document_store = self.initialize_sparse_docstore(self.sparse_index_name + "_" + sparse_type)
        sparse_retriever = self.initialize_sparse_retriever(self.sparse_type, sparse_document_store)
        self.pipelines[sparse_type] = DocumentSearchPipeline(sparse_retriever)

    def set_up_sparse_pipelines(self, recreate_index: bool = False) -> (DocumentSearchPipeline, DocumentSearchPipeline):
        """
        :param recreate_index:  Whether to recreate the ElasticSearch index from scratch (default=False). Useful for
                                debugging.
        """
        sparse_document_store_plain = self.initialize_sparse_docstore(self.sparse_index_name + "_bm25")
        sparse_document_store_multimatch = self.initialize_sparse_docstore(self.sparse_index_name + "_bm25f")

        # TODO: very first step; check if index already exists and then simply load it, without initialising models
        if sparse_document_store_plain.client.indices.exists(self.sparse_index_name + "_bm25") and \
                sparse_document_store_plain.client.indices.exists(self.sparse_index_name + "_bm25f") and \
                not recreate_index:
            # don't load up the clusterer and SPaR
            logger.info('[DocumentStore] Using existing indices')
        else:
            # prepare documents to be stored in the document store
            self.prepare_pipeline_inputs()
            # write documents to document store
            documents_to_write = []
            processed_document_fps = glob.glob(self.converted_output_dir + "*.json")
            logger.info("[DocumentStore] converting to ElasticSearch indexable passages.")
            for doc_fp in tqdm(processed_document_fps):
                documents_to_write += CustomDocument.load_document(doc_fp).to_list_of_dicts()
            logger.info(
                '[DocumentStore] Index and classifier_data will be created from scratch -- this will take a long time, \n'
                '                but only has to be done once.')
            sparse_document_store_plain.write_documents(documents_to_write)
            sparse_document_store_multimatch.write_documents(documents_to_write)

        # set up retriever with our document sto and return pipeline
        bm25_retriever = self.initialize_sparse_retriever("plain", sparse_document_store_plain)
        bm25f_retriever = self.initialize_sparse_retriever("multi_match", sparse_document_store_multimatch)
        return DocumentSearchPipeline(bm25_retriever), DocumentSearchPipeline(bm25f_retriever)

    def set_up_dense_pipeline(self,
                              field_to_index: str = "context",
                              recreate_index: bool = False) -> DocumentSearchPipeline:
        """
        Set up the various Retrieval nodes that are defined in the configuration.
        """
        # Prepare filepaths for storing the FAISS index
        faiss_index_path = f'/data/indexes/faiss/{self.conversion}_{field_to_index}_index'
        faiss_sql_doc_store = f'/data/indexes/faiss/{self.conversion}_{field_to_index}_document_store.db'

        # Prepare FAISS DocumentStore for dense indexing
        if os.path.exists(faiss_index_path) and not recreate_index:
            save_updated_document_store = False
            dense_document_store = FAISSDocumentStore.load(index_path=faiss_index_path)
        else:
            save_updated_document_store = True

            # Preprocessing inputs where necessary
            self.prepare_pipeline_inputs()

            # Create the document store
            sql_doc_store = f"sqlite:///{faiss_sql_doc_store}"
            dense_document_store = FAISSDocumentStore(embedding_dim=self.hidden_dims,
                                                      faiss_index_factory_str=self.faiss_index_type,
                                                      sql_url=sql_doc_store,
                                                      duplicate_documents="overwrite",  # instead skip?
                                                      similarity="dot_product")

            # write the processed documents to the Documentstore
            documents_to_write = []
            processed_document_fps = glob.glob(self.converted_output_dir + "*.json")
            for doc_fp in processed_document_fps:
                flat_content_list = CustomDocument.load_document(doc_fp).to_flat_list_of_dicts()

                # Avoid redundancy of X number of entries with the same doc_title
                if field_to_index == "doc_title":
                    flat_content_list[0]['content'] = flat_content_list[0]["doc_title"]
                    documents_to_write.append(flat_content_list[0])
                    continue

                # Change the content to be indexed based on field_to_index
                for c in flat_content_list:
                    c['content'] = c[field_to_index]

                documents_to_write += flat_content_list

            logger.info("[DENSE] Writing {} to dense document store".format(field_to_index))
            dense_document_store.write_documents(documents_to_write)

        # -- Dense Retriever(FAISS) https://docs.haystack.deepset.ai/reference/document-store-api#faissdocumentstore
        retriever = DensePassageRetriever(
            document_store=dense_document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            max_seq_len_query=self.max_seq_len_query,
            max_seq_len_passage=self.max_seq_len_passage,
            batch_size=self.batch_size,
            use_gpu=self.use_gpu,
            embed_title=True,
            use_fast_tokenizers=True
        )
        # Seems like we always have to update the embeddings after the DPR retriever init (takes a couple of minutes)
        # - https://haystack.deepset.ai/tutorials/dense-passage-retrieval
        logger.info("[Document Store] updating retriever embeddings for field:", field_to_index)
        dense_document_store.update_embeddings(retriever)

        if save_updated_document_store:
            # Save the document_store for reloading
            dense_document_store.save(index_path=faiss_index_path)

        return DocumentSearchPipeline(retriever)

    def run_query(self,
                  query="test for heat barrier in a roof"):
        # Get the predictions for each pipeline
        all_predictions = {}
        if self.indexing_type in ['sparse', 'hybrid']:
            # Currently using multimatch BM25; can also run BM25F in query_sparse()
            sparse_type = "bm25f" if self.sparse_type == "bm25f" else "bm25"
            all_predictions['bm25'] = self.pipelines[sparse_type].run(query=query)

        if self.indexing_type in ['dense', 'hybrid']:
            for field_to_index in self.pipelines_to_query:
                all_predictions[field_to_index] = self.pipelines[field_to_index].run(query=query, params={
                                                      "Retriever": {"top_k": self.top_k_per_retriever}
                                                  })
        return all_predictions
