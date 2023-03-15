from typing import List, Union
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier, NearestNeighbors

from embedder import Embedder


class Classifier:
    def __init__(self,
                 embedder: Embedder,
                 foreground_term_filepath: Path,
                 background_term_filepath: Path,
                 top_k_semantic_similarity: int = 5,
                 metric: str = 'euclidean'
                 ):
        """
        Handler for training a KNeighborsClassifier and using it for predictions on unseen spans. Init provides
        access to the Embedder specified in the settings, as well as corresponding data and filtered SPaR.txt outputs.

        :param embedder:     Embedder object that was used to create embeddings as determined by the settings file.
        :param foreground_term_filepath:   Path to file with all the cleaned SPaR.txt outputs from the foreground
                                           corpus.
        :param background_term_filepath:   Path to file with all the cleaned SPaR.txt outputs from the background
                                           corpus.
        :param top_k_semantic_similarity:   This is n_neighbours, the number of neighbours we compute for each term in
                                            the semantic similarity knn_graph.
        :param metric:   Distance metric to use for (1) the `KNeighborsClassifier`, (2) the `kneighbors_graph` that
                         is used to compute domain-classification heuristics, and (3) the `kneighbors_graph` used to
                         compute the top_k semantically similar terms.
        """

        self.embedder = embedder
        self.standardised_embedding_data = embedder.standardised_embedding_data
        self.foreground_terms = pickle.load(open(foreground_term_filepath, 'rb'))
        self.background_terms = pickle.load(open(background_term_filepath, 'rb'))

        self.domain_terms = None
        self.ood_terms = None
        self.top_k_semantic_similarity = top_k_semantic_similarity
        self.nearest_neighbours = None
        self.unique_spans = []
        self.knn_sim_dict = {}

        self.metric = metric

        # todo; pickle the classifier after training, so we can re-use it
        # todo; pickle the classifier after training, so we can re-use it
        # todo; pickle the classifier after training, so we can re-use it
        # todo; pickle the classifier after training, so we can re-use it
        # todo; pickle the classifier after training, so we can re-use it
        self.knn_classifier = KNeighborsClassifier(n_neighbors=50,
                                                   weights='distance',
                                                   leaf_size=100,
                                                   metric=metric,
                                                   n_jobs=4)

    def train_classifier_from_heuristics(self,
                                         classifier_nr_neighbours: int = 500,
                                         min_tfidf_value: float = 0.6,
                                         min_num_foreground_neighbours: int = 375
                                         ):
        """

        :param classifier_nr_neighbours:    The number of neighbours we compute for each term.
        :param min_tfidf_value:    The minimum value for the TFIDF-based heuristic to determine domain-specificity.
        :param min_num_foreground_neighbours:    The minimum number of neighbours from the foreground corpus for a span
                                                 to be considered within-domain -- the second heuristic.
        """
        # check if classifier exists;
        classifier_path = self.embedder.embedding_dir.joinpath("classifier.pkl")
        if classifier_path.exists():
            self.knn_classifier = pickle.load(open(classifier_path, 'rb'))
            self.domain_terms = pickle.load(open(self.embedder.embedding_dir.joinpath('domain_terms.pkl'), 'rb'))
            self.ood_terms = set(pickle.load(open(self.embedder.embedding_dir.joinpath('ood_terms.pkl'), 'rb')))

        # (1) compute the knn graph for the standardised embedding data [(span, embedding), (span, embedding), ...]
        knn_graph = kneighbors_graph(self.standardised_embedding_data,
                                     classifier_nr_neighbours,
                                     metric=self.metric,
                                     n_jobs=8)

        # (2) compute the number of background neighbours and foreground neighbours as a feature
        span_df_dict = {}
        for span_idx, (span, _) in enumerate(self.standardised_embedding_data):
            number_of_background_corpus_neighbours = 0
            for neighbour_idx in knn_graph[span_idx].indices:
                if self.standardised_embedding_data[neighbour_idx][0] in self.ood_terms:
                    number_of_background_corpus_neighbours += 1

            span_df_dict[span] = {
                'span_idx': span_idx,
                'num_background_neighbours': number_of_background_corpus_neighbours,
                'num_foreground_neighbours': classifier_nr_neighbours - number_of_background_corpus_neighbours
            }

        # (3) compute a TF-IDF style feature, inspired by Meyers et al. (2018)
        cleaned_foreground_terms_c = Counter(self.foreground_terms)
        cleaned_background_terms_c = Counter(self.background_terms)

        for span, _ in self.standardised_embedding_data:
            foreground_cnt = cleaned_foreground_terms_c[span]
            background_cnt = cleaned_background_terms_c[span]
            TF_fore_back = np.log(1 + (foreground_cnt / (foreground_cnt + background_cnt)))

            tokens, indices = self.embedder.prepare_tokens(span)
            idf_weights = self.embedder.get_idf_weights_for_indices(tokens, indices)
            TFIDF_fore_back = TF_fore_back * np.log(np.mean(idf_weights))

            span_df_dict[span]['TFIDF_fore_back'] = TFIDF_fore_back

        # (4) Pandas DF with all features
        span_features_df = pd.DataFrame.from_dict(span_df_dict, orient='index')
        domain_terms_df = span_features_df[(span_features_df['TFIDF_fore_back'] >= min_tfidf_value) & (
                          span_features_df['num_foreground_neighbours'] >= min_num_foreground_neighbours)]
        ood_terms_df = span_features_df[(span_features_df['TFIDF_fore_back'] < min_tfidf_value) | (
                       span_features_df['num_foreground_neighbours'] < min_num_foreground_neighbours)]

        # (5) write the domain and Out Of Domain (ood) terms to file, store in embeddings directory
        self.domain_terms = list(set(domain_terms_df.index.tolist()))
        self.ood_terms = list(set(ood_terms_df.index.tolist()))
        pickle.dump(self.domain_terms, open(self.embedder.embedding_dir.joinpath('domain_terms.pkl'), 'wb'))
        pickle.dump(self.ood_terms, open(self.embedder.embedding_dir.joinpath('ood_terms.pkl'), 'wb'))

        # (6) prepare training data for the classifier
        knn_spans, knn_X = zip(*self.standardised_embedding_data)
        knn_y = ['y' if s in self.domain_terms else 'n' for s in knn_spans]

        # set up the classifier and save it
        self.knn_classifier.fit(knn_X, knn_y)

    def predict_domains(self, spans_to_predict: Union[List[str], str] = None) -> List[str]:
        """
        Predict whether the `spans_to_predict` belongs to the foreground corpus domain.

        :param spans_to_predict: Spans for which you'd like to predict whether it belongs to the foreground corpus domain.
        :returns:   Returns the input spans that are classified as within domain.
        """
        domain_spans = []
        if spans_to_predict:
            if type(spans_to_predict) == str:
                spans_to_predict = [spans_to_predict]
            for span in spans_to_predict:
                normalised_embedding = self.embedder.embed_and_normalise_span(span).reshape(1, -1)
                if self.knn_classifier.predict(normalised_embedding)[0] == 'y':
                    domain_spans.append(span)
        return domain_spans

    def prep_semantic_similarity(self):
        """
        Prepares a dictionary that holds the top_k nearest neighbours for each span that is present in (1) the
        domain spans extracted from the foreground corpus, and (2) any domain-spans found in definitions of the KG.
        """
        all_embedding_data = self.embedder.standardised_embedding_data + self.embedder.new_embedding_data
        unique_spans, unique_embeddings = zip(*all_embedding_data)
        stacked_embeddings = np.stack([np.mean(e, axis=0) if len(e.shape) > 1 else e for e in unique_embeddings])
        knn_graph = kneighbors_graph(stacked_embeddings,
                                     self.top_k_semantic_similarity,
                                     metric=self.metric,
                                     n_jobs=8)

        for span_idx, span in enumerate(unique_spans):
            self.knn_sim_dict[span] = [unique_spans[neighbour_idx] for neighbour_idx in knn_graph[span_idx].indices]

    def get_semantically_similar_terms(self, span_list: Union[List[str], str] = None) -> List[List[str]]:
        """
        Simply get the `self.top_k_semantic_similarity` neighbours for each span in a list. Note that for a span to be
        assigned a set of semantically similar span, it has to have been considered a domain term in our classifier. If
        the span does not occur in `self.knn_sim_dict`, then it will be assigned an empty list of neighbours.
        """
        if not self.knn_sim_dict:
            self.prep_semantic_similarity()

        if type(span_list) == str:
            span_list = [span_list]

        neighbours_lists = []
        for span in span_list:
            if span in self.knn_sim_dict:
                neighbours_lists.append(self.knn_sim_dict[span])
            else:
                neighbours_lists.append([])

        return neighbours_lists

    def prep_nearest_neighbours(self):
        all_embedding_data = self.embedder.standardised_embedding_data + self.embedder.new_embedding_data
        self.unique_spans, unique_embeddings = zip(*all_embedding_data)
        stacked_embeddings = np.stack([np.mean(e, axis=0) if len(e.shape) > 1 else e for e in unique_embeddings])
        self.nearest_neighbours = NearestNeighbors(n_neighbors=self.top_k_semantic_similarity,
                                                   metric=self.metric,
                                                   algorithm='auto',
                                                   n_jobs=-1)
        self.nearest_neighbours.fit(stacked_embeddings)

    def get_nearest_neighbours(self, span_list: Union[List[str], str] = None) -> List[List[str]]:
        """
        Compute the `self.top_k_semantic_similarity` neighbours for each span in a list.
        """
        if not self.nearest_neighbours:
            self.prep_nearest_neighbours()

        if type(span_list) == str:
            span_list = [span_list]

        neighbours_lists = []
        for span in span_list:
            if span in self.knn_sim_dict:
                embedding = self.embedder.embed_and_normalise_span(span)
                neighbour_indices = self.nearest_neighbours.kneighbors(embedding,
                                                                       self.top_k_semantic_similarity,
                                                                       return_distance=False)
                neighbour_spans = [self.unique_spans[idx] for idx in neighbour_indices]
                neighbours_lists.append(neighbour_spans)
            else:
                neighbours_lists.append([])

        return neighbours_lists
