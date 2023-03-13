from typing import List, Union, Dict, Any
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier

from embedder import Embedder


class Classifier:
    def __init__(self,
                 embedder: Embedder,
                 foreground_term_filepath: Path,
                 background_term_filepath: Path,
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
        self.knn_sim_dict = {}

        self.metric = metric
        self.knn_classifier = KNeighborsClassifier(n_neighbors=50,
                                                   weights='distance',
                                                   leaf_size=100,
                                                   metric=metric,
                                                   n_jobs=4)

    def train_classifier_from_heuristics(self,
                                         n_neighbours: int = 500,
                                         min_tfidf_value: float = 0.6,
                                         min_num_foreground_neighbours: int = 250
                                         ):
        """

        :param n_neighbours:    The number of neighbours we compute for each term.
        :param min_tfidf_value:    The minimum value for the TFIDF-based heuristic to determine domain-specificity.
        :param min_num_foreground_neighbours:    The minimum number of neighbours from the foreground corpus for a span
                                                 to be considered within-domain -- the second heuristic.
        """
        # (1) compute the knn graph for the standardised embedding data [(span, embedding), (span, embedding), ...]
        knn_graph = kneighbors_graph(self.standardised_embedding_data,
                                     n_neighbours,
                                     metric=self.metric,
                                     n_jobs=8)

        # (2) compute the number of background neighbours and foreground neighbours as a feature
        span_df_dict = {}
        for span_idx, (span, _) in enumerate(self.standardised_embedding_data):
            number_of_background_corpus_neighbours = 0
            for neighbour_idx in knn_graph[span_idx].indices:
                if self.standardised_embedding_data[neighbour_idx][0] in self.cleaned_background_terms_c.keys():
                    number_of_background_corpus_neighbours += 1

            span_df_dict[span] = {'span_idx': span_idx,
                                  'num_background_neighbours': number_of_background_corpus_neighbours,
                                  'num_foreground_neighbours': n_neighbours - number_of_background_corpus_neighbours}

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

        self.domain_terms = list(set(domain_terms_df.index.tolist()))
        self.ood_terms = list(set(ood_terms_df.index.tolist()))

        # (5) write the unique domain terms (spans) to file, store in embeddings directory
        pickle.dump(self.domain_terms, open(self.embedder.embedding_dir.joinpath('domain_terms.pkl'), 'wb'))

        # (6) prepare training data for the classifier
        knn_spans, knn_X = zip(*self.standardised_embedding_data)
        knn_y = ['y' if s in self.domain_terms else 'n' for s in knn_spans]

        # set up the classifier
        self.knn_classifier.fit(knn_X, knn_y)

    def predict(self, span_to_predict: str) -> str:
        """
        Predict whether the `span_to_predict` belongs to the foreground corpus domain.

        :param span_to_predict: Span for which you'd like to predict whether it belongs to the foreground corpus domain.
        :returns:   Returns either 'y' or 'n'.
        """
        if span_to_predict:
            return self.knn_classifier.predict(self.embedder.embed_and_normalise(span_to_predict).reshape(1, -1))[0]
        else:
            return 'n'

    def compute_semantic_similarity(self,
                                    top_k: int = 5
                                    ) -> List[str]:
        """

        :param top_k:   This is n_neighbours, the number of neighbours we compute for each term in the knn_graph.
        """
        all_embedding_data = self.embedder.standardised_embedding_data + self.embedder.new_embedding_data
        unique_spans, unique_embeddings = zip(*all_embedding_data)
        stacked_embeddings = np.stack([np.mean(e, axis=0) if len(e.shape) > 1 else e for e in unique_embeddings])
        knn_graph = kneighbors_graph(stacked_embeddings,
                                     top_k,
                                     metric=self.metric,
                                     n_jobs=8)

        for span_idx, span in enumerate(unique_spans):
            self.knn_sim_dict[span] = [unique_spans[neighbour_idx] for neighbour_idx in knn_graph[span_idx].indices]

