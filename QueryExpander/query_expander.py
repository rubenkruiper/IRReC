from typing import List, Dict
import pickle
import requests
import numpy as np
import networkx as nx

from textblob import TextBlob
from collections import Counter

from query_expansion_utils import cleaning_helper, levenshtein


class QueryExpander:
    def __init__(self,
                 classifier_url: str,
                 ner_url: str,
                 prf_weight=0,
                 kg_weight=0,
                 nn_weight=0):
        self.ner_url = ner_url
        self.classifier_url = classifier_url

        self.prf_weight = prf_weight
        self.kg_weight = kg_weight
        self.nn_weight = nn_weight

        # todo - create this graph from the IR system; too much work for now
        self.network = pickle.load(open("/data/undirected_weighted_graph.pkl", 'rb'))
        self.avg_degree_dict = nx.average_neighbor_degree(self.network)
        self.graph_nodes = list(self.network.nodes)

    def expand_query(self,
                     initial_query: str,
                     initial_search_results: Dict = None):
        """

        :param initial_query:           Input query provided by a user.
        :return expanded_query:         Expanded version of the query (currently appending candidates).
        :param initial_search_results:   Regular search results to identify PRF candidates, only when using PRF
        """
        # run the different methods that identify candidates
        spans, nn_candidates, kg_candidates, prf_candidates = self.compute_candidates(initial_query,
                                                                                                 initial_search_results)

        top_candidates, qe_insight = self.score_and_select_candidates(nn_candidates,
                                                                      kg_candidates,
                                                                      prf_candidates,
                                                                      spans)
        expanded_query = self.append_spans_to_query(initial_query, spans, top_candidates)
        return expanded_query, qe_insight

    def compute_candidates(self,
                           initial_query: str,
                           initial_search_results: Dict[str, str]):
        # identify which query keywords belong together; currently expecting a sentence or punctuation to help splitting
        spans = self.span_identification_SPaR(initial_query)
        spans = cleaning_helper(spans)                               # run some basic cleaning on spans before QE

        # nearest neighbours of spans # todo; change the manual setting of top_k here
        nn_candidates = self.nearest_neighbours(spans, top_k=2) if self.nn_weight else None

        # # related spans found in domain vocabulary
        kg_candidates = self.span_KG_mapping(spans) if self.kg_weight else None

        # # PRF ~ spans found in documents that were ranked high based on the initial query
        if self.prf_weight:
            prf_candidates = self.pseudo_relevance_feedback(initial_search_results, spans)
        else:
            prf_candidates = None

        return spans, nn_candidates, kg_candidates, prf_candidates

    def clean_candidate(self, candidate):
        """ Some characters in candidates mess up the querying system"""
        candidate = candidate.replace('"', '').replace("'", '').replace('(', '').replace(')', '')
        candidate = candidate.replace('/', '').replace('\\', '')
        return candidate

    def score_and_select_candidates(self, nn_candidates,
                                    kg_candidates,
                                    prf_candidates,
                                    spans, top_k=5):
        """
        Select top X number of candidates; usually ranking is based on IDF values!
        """
        # ToDo; select not the top X, but a number of terms based on the initial query length?
        candidates_and_weights = {}
        if nn_candidates:
            candidates_and_weights["nearest_neighbours"] = {"candidates": nn_candidates,
                                                            "weight": self.nn_weight}
        if kg_candidates:
            candidates_and_weights["kg_candidates"] = {"candidates": kg_candidates,
                                                       "weight": self.kg_weight}
        if prf_candidates:
            candidates_and_weights["pseudo_relevant_terms"] = {
                "candidates": prf_candidates,
                "weight": self.prf_weight}

        score_counter = Counter()
        qe_insight = {'spans': spans}
        for k, v in candidates_and_weights.items():
            current_candidates_counter = Counter()
            qe_insight[k] = {'weight': v['weight']}
            for idx, candidate in enumerate(v['candidates']):
                if candidate in spans:
                    continue
                # todo; speed up by computing idf weights for all candidates in 1 go perhaps
                response = requests.post(f"{self.classifier_url}get_idf_weights/", json={"spans": [candidate]})
                if response:
                    idf_weights = response.json()["idf_weights"][0]
                    # assuming average IDF weights over the tokens for now
                    weighted_avg_idf_weight = (sum(idf_weights)/len(idf_weights)) * v['weight']
                    score_counter[candidate] += weighted_avg_idf_weight
                    current_candidates_counter[candidate] += weighted_avg_idf_weight
                else:
                    score_counter[str(candidate)] += 0.0001
                    current_candidates_counter[str(candidate)] += 0.0001
            qe_insight[k]['candidates'] = current_candidates_counter.most_common()

        qe_insight['top_qe_terms'] = [c for c, _ in score_counter.most_common(top_k)]
        clean_candidates = [self.clean_candidate(c) for c, _ in score_counter.most_common(top_k)]
        return clean_candidates, qe_insight

    def append_spans_to_query(self, initial_query, spans, top_candidates):
        """ Simply append spans together to form a longer query with more potential for matching relevant results. """
        # ToDo; potentially combine the query spans and expansion candidates in a different way
        try:
            return initial_query + ", " + ", ".join(spans + top_candidates)
        except TypeError as e:
            raise Exception(e,
                            f"initial query contains: {type(initial_query[0])}\ncandidates: {type(top_candidates[0])}")

    def get_unique_words_in_query(self, query: str, candidates: List[str]):
        """
        For sparse IR, the query is treated as a bag-of-words and so the count of terms matters. This function simply
        combines the words from the query and the expansion candidates and returns a string containing each word once.
        """
        query_words = [w for w in TextBlob(query).words]
        for candidate in candidates:
            query_words.append([w for w in TextBlob(candidate).words])
        return ' '.join(set(query_words))

    def span_identification_SPaR(self, initial_query: str) -> List[str]:
        """
        Identify the spans that occur in the query.

        :param initial_query:   String that holds the original input query.
        :return spans:    List of objects (strings) identified in the query using SPaR.txt
        """
        response = requests.post(f"{self.ner_url}predict_objects/",
                                 json={"sentence": initial_query}).json()
        try:
            return response['prediction']['obj']
        except KeyError as e:
            raise Exception(e, f"NER labeling issue for {initial_query}\n -->{response}")

    def nearest_neighbours(self, spans: List[str], top_k: int = 3) -> List[str]:
        """
        identify nearest neighbours → candidate set 1
        As the input consists of a list of strings, the output would be a list of lists; I think
         -- A list of lists, each nested list holds neighbours of query spans
        we

        :param spans:   List of strings, representing objects that SPaR identified in the query
        :param top_k:   Max nr of neighbours to identify as candidate, per span.
        :return nearest_neighbours_for_spans:    List of strings, flattened from a list of lists -- where each nested
                                                 list holds neighbours of the input spans
        """

        response = requests.post(f"{self.classifier_url}get_neighbours/", json={"spans": spans}).json()
        dissimilar_neighbours = []
        for span, nn_list in zip(spans, response['neighbours']):
            dissimilar_neighbours += [nn for nn in nn_list if (
                    nn.lower() not in span.lower() and not levenshtein(nn, span)
            )][:top_k]
        return dissimilar_neighbours

    def span_KG_mapping(self, spans: List[str], minimum_degree: int = 75, top_k: int = 2) -> List[str]:
        """
        KG expansion → candidate set 2
        Based on the "distance" between nodes in our graph, grab the top_k QE candidates.

        :param spans:            List of spans.
        :param minimum_degree:   We will only expand nodes that are 'uncommon' in the graph, e.g.,
                                 a degree lower than this value.
        :param top_k:            Number of QE candidates we will return, per span again (like with NNs).
        :return:                 List of candidate spans
        """
        # check if the query-derived span exists as a node in the network
        # todo if the node doesn't exit in the network, use a back-off strategy maybe
        query_nodes = [n for n in spans if n in self.graph_nodes]

        kg_neighbour_candidates = []
        # expanded_nodes_dict = {}  # todo consider grouping QE candidates per span?
        dist_weight = 1             # todo consider using weights in changing KG-based QE results
        degree_weight = 1
        ego_radius = 5
        for node in query_nodes:
            if node not in self.network:
                # we already filter out spans that do not exist in the network before, here we simply avoid errors
                # todo; clean -- probably keep this and not the self.graph_nodes comparison
                continue

            own_degree = self.network.degree[node]

            if own_degree > minimum_degree:
                # We would primarily like to use QE for terms that are not very common
                continue

            # we will look at the ego_graph neighbours
            ego_graph = nx.ego_graph(self.network, node, radius=ego_radius)
            kg_neighbours = []
            for n in ego_graph.neighbors(node):
                if self.avg_degree_dict[n] > minimum_degree:
                    # We would like to use QE to find terms that are not extremely common
                    continue

                distance = self.network.get_edge_data(node, n)
                degree_measure = np.log(self.network.degree[n]) + np.log(self.avg_degree_dict[n])
                # We'll assume that avg degree represents the connectedness of the node, as well as the nodes it is
                # connected too. Distance represents the
                # -- degree (how common the term is in the KG)  --> lower degree is less common
                # -- distance (higher distance is worse) --> we would like a small distance between nodes
                combination_tuple = [(dist_weight * distance['weight']) + (degree_measure * degree_weight), n]
                kg_neighbours.append(combination_tuple)

            kg_neighbour_candidates.append([n for distance_degree, n in sorted(kg_neighbours)])

        dissimilar_kg_candidates = []
        for s, kg_candidate_list in zip(query_nodes, kg_neighbour_candidates):
            for c in kg_candidate_list:
                if s in c:
                    dissimilar_kg_candidates.append(c)
                elif (not levenshtein(s, c)) and (c.lower() not in s.lower()):
                    dissimilar_kg_candidates.append(c)

        return dissimilar_kg_candidates[:top_k]

    def pseudo_relevance_feedback(self, initial_search_results, spans, top_k: int = 2):
        """
        PRF - identify which filtered domain terms occur in initial retrieved docs → candidate set 3
        :input initial_search_results:  This is the `combined_pred` dict that the API's regular query function returns.
        """
        prf_candidates = []
        for idx, score_and_dict in enumerate(initial_search_results):
            score, d = score_and_dict
            label_counter = d['label_counters_dict']['filtered_NER_labels']     # can change label here, or use multiple
            filtered_NER_labels = [(str(r), int(c)) for r, c in label_counter.most_common()]
            prf_candidates += filtered_NER_labels   # can combine multiple sources of labels

        prf_counter = Counter()
        for r, c in prf_candidates:
            prf_counter[r] += c

        # we grab the top_k most common candidate first
        prf_labels = [label for label, count in prf_counter.most_common() if (label != '' and label not in spans)]
        prf_counts = [count for label, count in prf_counter.most_common() if (label != '' and label not in spans)]

        dissimilar_prf_candidates = []
        for count, label in zip(prf_counts, prf_labels):
            if count < 2:
                # let's assume we want each candidate to occur at least twice in all the retrieved documents
                continue

            sims = [False if (not levenshtein(label, s) and not label.lower() in s.lower()) else True for s in spans]

            if not any(sims):
                dissimilar_prf_candidates.append(label)

            if len(dissimilar_prf_candidates) >= top_k:
                break

        return dissimilar_prf_candidates[:top_k]
