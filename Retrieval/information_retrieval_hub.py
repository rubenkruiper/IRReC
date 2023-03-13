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
from collections import Counter
from typing import List, Dict
from pathlib import Path

# my utilities
from Retrieval.utils import spar_utils, cleaning_utils
from Retrieval.utils.IDF_computer import IdfComputer
from Retrieval.utils.customdocument import CustomDocument
from Retrieval.utils.conversion_utils import convert_inputs
from Retrieval.utils.preprocessor_spartxt import SparPreProcessor

# Haystack imports
from haystack.document_stores import ElasticsearchDocumentStore, FAISSDocumentStore
from haystack.nodes import ElasticsearchRetriever, DensePassageRetriever
from haystack.pipelines import DocumentSearchPipeline

logger = logging


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

        # urls for SPaR and Classifier APIs
        self.ner_url = configs["retrieval"]["ner_url"]
        self.classifier_url = configs["retrieval"]["clustering_url"]

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
        self.preprocess_inputs(self.foreground_output_dir, self.classifier_dir.joinpath("foreground_terms.pkl"))
        self.preprocess_inputs(self.background_output_dir, self.classifier_dir.joinpath("background_terms.pkl"))

        # (4) SPaR.txt is done, at this stage we want to prepare the domain classification
        # do I want to pass specific settings here? json= {'num_neighbours':x, etc...}
        requests.post(f"{self.classifier_url}train/")

        # (5) Once the classifier is trained, prep the KG?


    def preprocess_inputs(self,
                          converted_documents_dir: Path,
                          term_output_path: Path = Path("/data/classifier_data/foreground_terms.pkl"),
                          split_length: int = 100) -> List[CustomDocument]:
        """
        Takes the text found in pdfs during the conversion step, splits it into passages and
        reformats for indexing.

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
            print("[Preprocessor] Running SPaR.txt to collect potential NER labels.")
            for converted_document_filepath in tqdm(converted_document_filepaths):
                converted_document = CustomDocument.load_document(converted_document_filepath)
                if not converted_document:
                    raise Exception(f"Issue loading converted document: {converted_document_filepath}")

                if any([len(c.NER_labels) > 1 for c in converted_document.all_contents]):
                    logger.info(f"[Preprocessor] skipping, NER outputs already found in: {converted_document_filepath}")
                else:
                    logger.info(f"[Preprocessor] Preprocessing classifier_data for: {converted_document_filepath}")
                    processed_list_of_dicts = self.preprocessor.process(converted_document.to_list_of_dicts())
                    converted_document.replace_contents(processed_list_of_dicts)
                    converted_document.write_document()
                # keep track of all the NER labels
                all_NER_labels += [label for c in converted_document.all_contents for label in c.NER_labels]

            # group all the (cleaned) terms extracted for a corpus and store for classification later
            cleaned_terms = self.preprocessor.cleaning_helper(all_NER_labels)
            with open(term_output_path, 'wb') as f:
                pickle.dump(f, cleaned_terms)
            filtered_labels = set(cleaned_terms)
        else:
            filtered_labels = set(pickle.load(open(term_output_path, 'rb')))

        # (2) identify which NER labels remain after cleaning
        print("[Preprocessor] Retaining the filtered/cleaned SPaR.txt labels separately.")
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
                        processed_document.all_contents[idx].set_filtered_NER_labels(only_keep_filtered)
                # update stored file
                processed_document.write_document()

    def identify_similar_labels(self,
                                converted_documents_dir: Path,
                                term_output_path: Path = Path("/data/classifier_data/foreground_terms.pkl")):
        """
        We want to keep track of the filtered and classified terms in a CustomDocument's contents. This simply updates
        the converted document at it's file location.

        :param converted_documents_dir:    The path where we can find out converted documents.
        :param term_output_path:    The path where all the filtered terms for this corpus are stored.
        """
        preprocessed_document_filepaths = glob.glob(self.converted_documents_dir + '*json')


        # (2) identify the X most similar labels, IF within domain?
        if self.classifier_url not in ["no", "No", "None", "none", ""]:
            if not self.clusters_to_filter:
                self.clusters_to_filter = requests.get(self.classifier_url +
                                                       "get_clusters_to_filter/").json()["clusters_to_filter"]

            for fp in tqdm(preprocessed_document_fps):
                processed_document = CustomDocument.load_document(fp)
                if any([len(c.cluster_neighbours) > 1 for c in processed_document.all_contents]):
                    logger.info(
                        "[Cluster Labels] Skipping cluster labelling, since cluster labels found in: {}".format(
                            processed_document.output_fp
                        ))
                else:
                    # We want to assign a cluster to all filtered NER labels in a document (collection of passages)
                    content_and_labels_tracking_dict = {}
                    num_unique_label_ids = 0
                    doc_labels = []
                    for idx, content in enumerate(processed_document.all_contents):
                        current_content_labels = content.filtered_NER_labels
                        doc_labels += current_content_labels
                        content_and_labels_tracking_dict[idx] = [x for x in range(num_unique_label_ids,
                                                                                  num_unique_label_ids +
                                                                                  len(current_content_labels))]
                        num_unique_label_ids += len(current_content_labels)

                    # We assign in batches of size num.clusters; don't remember why, I thought kmcuda runs into issues
                    num_clusters = requests.get(self.classifier_url + "get_num_clusters/").json()["num_clusters"]
                    doc_label_splits = split_list(doc_labels, num_clusters)
                    doc_cluster_ids = []
                    doc_cluster_neighbours = []
                    for objects_to_assign_id in doc_label_splits:
                        # Assign a cluster ID and identify close neighbours for the objects in a give slice
                        # todo - ability to change the cosine sim threshold and the nr of results to be stored
                        params = {"data": objects_to_assign_id,
                                  "cosine_sim_threshold": 0.7,
                                  "max_results": 3}
                        response = requests.post(self.classifier_url + "get_neighbours/", json=params).json()
                        doc_cluster_ids += response['assigned_ids']
                        doc_cluster_neighbours += response['neighbours']

                    # Now we make sure the ids and neighbours are assigned to the right contents again.
                    for idx, label_ids in content_and_labels_tracking_dict.items():
                        nested_cluster_ids = [i for i in doc_cluster_ids[label_ids[0]:label_ids[-1] + 1] if i != -1]
                        if not nested_cluster_ids:
                            logger.info(f"[Clustering] Issue with assigned IDs: {nested_cluster_ids}")
                            assigned_neighbours = []
                            assigned_cluster_ids = []
                        elif any(type(cid) != int for cid in nested_cluster_ids):
                            # expecting a nested list [[1,2,3],[4,5,...]]
                            assigned_cluster_ids = list(itertools.chain(*nested_cluster_ids))
                            nested_neighbours = doc_cluster_neighbours[label_ids[0]:label_ids[-1] + 1]
                            assigned_neighbours = list(itertools.chain(*nested_neighbours))
                        else:
                            # expecting a list of assigned cluster ids [1,2,3,...]
                            assigned_cluster_ids = nested_cluster_ids
                            assigned_neighbours = doc_cluster_neighbours[label_ids[0]:label_ids[-1] + 1]

                        processed_document.all_contents[idx].set_cluster_neighbours(assigned_neighbours)

                        filtered_neighbours = []
                        for n, a in zip(assigned_neighbours, assigned_cluster_ids):
                            if a not in self.clusters_to_filter:
                                filtered_neighbours.append(n)

                        processed_document.all_contents[idx].set_cluster_filtered(filtered_neighbours)

                    # update stored file
                    processed_document.write_document()


    def initialize_sparse_docstore(self, index_name: str):
        fields = ["content", "doc_title", "SPaR_labels", "filtered_SPaR_labels",
                  "cluster_filtered", "cluster_neighbours"]

        # 'skip', 'overwrite' or 'fail'
        duplicate_documents = 'overwrite' if self.recreate_index else 'skip'

        # Start Elasticsearch using Docker via the Haystack utility function
        sparse_document_store = ElasticsearchDocumentStore(host="host.docker.internal",
                                                           index=index_name,
                                                           search_fields=fields,
                                                           content_field="content",
                                                           recreate_index=recreate_index,
                                                           duplicate_documents=duplicate_documents)
        return sparse_document_store

    def initialize_sparse_retriever(self, sparse_type, sparse_document_store: ElasticsearchDocumentStore):
        if sparse_type == "bm25f":           # "combined_fields"]: dropped for now
            fields = [f"content^{self.fields_and_weights['content']}",
                      f"doc_title^{self.fields_and_weights['doc_title']}",
                      f"SPaR_labels^{self.fields_and_weights['SPaR_labels']}",
                      f"filtered_SPaR_labels^{self.fields_and_weights['filtered_SPaR_labels']}",
                      f"cluster_filtered^{self.fields_and_weights['cluster_filtered']}",
                      f"cluster_neighbours^{self.fields_and_weights['cluster_neighbours']}"]

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
        sparse_document_store_plain = self.initialize_sparse_docstore(self.sparse_index_name + "_bm25",
                                                                      recreate_index)
        sparse_document_store_multimatch = self.initialize_sparse_docstore(self.sparse_index_name + "_bm25f",
                                                                           recreate_index)

        # TODO: very first step; check if index already exists and then simply load it, without initialising models
        if sparse_document_store_plain.client.indices.exists(self.sparse_index_name + "_bm25") and \
                sparse_document_store_plain.client.indices.exists(self.sparse_index_name + "_bm25f") and \
                not recreate_index:
            # don't load up the clusterer and SPaR
            print('[DocumentStore] Using existing indices')
        else:
            # prepare documents to be stored in the document store
            self.prepare_pipeline_inputs()
            # write documents to document store
            documents_to_write = []
            processed_document_fps = glob.glob(self.converted_output_dir + "*.json")
            print("[DocumentStore] converting to ElasticSearch indexable passages.")
            for doc_fp in tqdm(processed_document_fps):
                documents_to_write += CustomDocument.load_document(doc_fp).to_list_of_dicts()
            print(
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
            self.convert_inputs()
            # todo ; decide if I want to pass split_length here (just sticking to 100 for now)
            self.preprocess_inputs()
            self.filtered_and_classified()

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

        # -- Dense Retriever(FAISS)
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
