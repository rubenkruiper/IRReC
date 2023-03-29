# import math, re
from typing import List, Dict, Any
#
# from Levenshtein import distance as lev
# from scipy.spatial.distance import cosine, euclidean
#
# from embedder import Embedder


def split_list(some_list: List, chunk_size: int) -> List[List]:
    """
    Helper function to split a list into smaller lists of a given size.

    :param some_list:   List that has to be split into chunks.
    :param chunk_size:  Size of the sublists that will be returned.
    :return list_of_sublists:  A list of sublists, each with a maximum size of `chunk_size`.
    """
    return [some_list[i:i + chunk_size] for i in range(0, len(some_list), chunk_size)]


def subword_insight(subword_units: List[str],
                    weights: List[float],
                    idf_threshold: float,
                    text: str):
    """
    Function that prints out the text, as well as its tokens and corresponding weights. The tokens will be
    represented with strike-through if they weight is below the set idf_threshold.
    """
    print(f"Text: {text}")
    for subword_unit, w in zip(subword_units, weights):
        if w < idf_threshold:
            subword_unit = ''.join([str(c) + '\u0336' for c in subword_unit])
        print("{:.2f}\t{}".format(w, subword_unit))


def predict_uniques_all_contents(list_of_lists, function):
    flat_spans, list_ids = map(list, zip(*[(s, idx) for idx, spans in enumerate(list_of_lists) for s in spans]))
    unique_spans = list(set(flat_spans))
    print("\n\n[UNIQUE SPANS] :", unique_spans)
    computed_values = function(unique_spans)
    print("\n\n[COMPUTE VALUES] :", computed_values)
    dict_of_lists_to_return = {idx: [] for idx in list_ids}
    for span, idx in zip(flat_spans, list_ids):
        try:
            dict_of_lists_to_return[idx].append(computed_values[unique_spans.index(span)])
        except ValueError:
            # this happens when the function is domain classification, and the span is regarded as out-of-domain
            dict_of_lists_to_return[idx].append('')         # todo adding empty string for now
    # make sure to return the lists sorted by the key indices
    return [dict_of_lists_to_return[k] for k in sorted(dict_of_lists_to_return, key=dict_of_lists_to_return.get)]

#
# ############################################################################################################
# ###### classifier_data containers [deprecated?]
# class ToBeClustered:
#     def __init__(self, text: str, embedder: Embedder):
#         """
#         Object that holds a text-span and provides access to embedding-related classifier_data; tokenized_text,
#         token_indices, IDF_values_for_tokens, embedding.
#         """
#         self.text = text
#
#         self.tokenized_text, self.token_indices = embedder.prepare_tokens(text)
#         self.IDF_values_for_tokens = embedder.get_idf_weights_for_indices(self.tokenized_text, self.token_indices)
#         self.embedding = embedder.combine_token_embeddings(embedder.embed_text(text))
#
#         # placeholders for cluster_ID and neighbours
#         self.cluster_id = -1
#         self.distance_to_centroid = math.inf
#         self.all_neighbours = []    # TODO; change to kNN graph approach!
#
#         self.idf_threshold = embedder.idf_threshold
#
#     def print_tokens_and_weights(self, idf_threshold: float = None):
#         """ Aim is to check how the IDF threshold affects the influence of subword tokens on the entire span weight. """
#         if not idf_threshold:
#             idf_threshold = self.idf_threshold
#
#         subword_insight(self.tokenized_text, self.IDF_values_for_tokens, self.text, idf_threshold)
#
#     def get_top_k_neighbours(self, unique_span_dict: Dict[str, Any], cosine_sim_threshold: float = 0.7, top_k: int = 3):
#         """
#         Function to compute the `top_k` terms in a cluster that are closest to it's centroid. The idea is to compute
#         this list for the spans in a passage to-be-retrieved (before indexing) and for a query (during query-expansion).
#         The aim is thus to increase the similarity between query and document at the level of related terms.
#         Cosine similarity is used to further ensure embedding similarity, and levenshtein distance is used to make sure
#         the neighbours aren't simply the plural form.
#
#         :param unique_span_dict:   Dict with unique spans as keys and their embeddings as values, original input to
#                                    the KMeans clustering.
#         :param cosine_sim_threshold:   Minimum cosine similarity value for a neighbour to be considered 'similar'.
#         :param top_k:   An `int` value to determine the maximum number of related strings to return.
#         :return all_top_terms:  A `list` of terms from the assigned cluster.
#         """
#         # TODO; change to kNN graph approach!
#         # TODO; change to kNN graph approach!
#         # TODO; change to kNN graph approach!
#
#
#         # List that will hold the neighbours. Each new candidate will be compared to the spans in this list, so we
#         # include the original span itself as well for that comparison and later omit it.
#         all_top_terms = [self.text]
#         # sort neighbours (takes a long time but seems to work well... maybe I should try hierarchically cluster or smt)
#         #  - euclidean distance for now;
#         self.all_neighbours.sort(key=lambda x: euclidean(unique_span_dict[x[1]], self.embedding))
#
#         for dist, neighbour in self.all_neighbours:
#
#             # If the Levenshtein distance to other already added neighbours is too close (True), skip this neighbour
#             if any([levenshtein(already_added, neighbour) for already_added in all_top_terms]):
#                 continue
#
#             # Get the embedding for this neighbour span
#             try:
#                 neighbour_emb = unique_span_dict[neighbour]
#             except KeyError:
#                 # are these the neighbours that would/should be filtered?
#                 print(f"[Clustering] Potential neighbour '{neighbour}' no embedding found in unique_span_dict")
#                 continue
#
#             # If the cosine similarity is below a given threshold, discard
#             if (1 - cosine(self.embedding, neighbour_emb)) < cosine_sim_threshold:
#                 continue
#
#             all_top_terms.append(neighbour)
#             if len(all_top_terms) > top_k:
#                 break
#
#         return all_top_terms[1:]
#
