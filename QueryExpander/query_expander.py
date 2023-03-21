import requests
from urllib import parse
from textblob import TextBlob

from typing import List, Dict
from collections import Counter
from SPARQLWrapper import SPARQLWrapper, JSON

from sparql_utils import get_results_for_queries


# https://www.elastic.co/guide/en/elasticsearch/reference/7.17/analysis-synonym-graph-tokenfilter.html


class QueryExpander:
    def __init__(self,
                 sparql_endpoint: str,
                 classifier_url: str,
                 spar_url: str,
                 prf_weight=0,
                 kg_weight=0,
                 nn_weight=0,
                 bm25_weight=1):
        # todo - allow a user to set which candidates to collect?
        self.spar = spar_url
        self.classifier = classifier_url

        self.prf_weight = prf_weight
        self.kg_weight = kg_weight
        self.nn_weight = nn_weight
        self.bm25_weight = bm25_weight

        # set up connection to GraphDB sparql endpoint and retrieve a list of nodes (concepts in the graph)
        try:
            self.sparql = SPARQLWrapper(sparql_endpoint)
            self.sparql.setQuery("""
                                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                                PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                                select ?object where { 
                                    ?s ?p ?object .
                                    FILTER(?p = skos:prefLabel)
                                } """)
            self.sparql.setReturnFormat(JSON)
            all_prefLabel_nodes = self.sparql.query().convert()['results']['bindings']
            self.lowercased_nodes = set([n['object']['value'].lower() for n in all_prefLabel_nodes])
        except:
            print("Issue connecting to the provided SPARQL endpoint: ", sparql_endpoint)
            self.lowercased_nodes = []

    def expand_query(self,
                      initial_query: str,
                      initial_search_results: Dict = None):
        """

        :param initial_query:           Input query provided by a user.
        :return expanded_query:         Expanded version of the query (currently appending candidates).
        :param initial_search_results:   Regular search results to identify PRF candidates, only when using PRF
        """
        # run the different methods that identify candidates
        spans, nn_candidates, kg_candidates, prf_candidates_and_counts = self.compute_candidates(initial_query,
                                                                                                 initial_search_results)

        top_candidates, qe_insight = self.score_and_select_candidates(nn_candidates,
                                                                      kg_candidates,
                                                                      prf_candidates_and_counts,
                                                                      spans)
        expanded_query = self.append_spans_to_query(initial_query, spans, top_candidates)
        return expanded_query, qe_insight

    def compute_candidates(self,
                           initial_query: str,
                           initial_search_results: Dict[str, str]):
        # identify which query keywords belong together; currently expecting a sentence or punctuation to help splitting
        spans = self.span_identification_SPaR(initial_query)

        # nearest neighbours of spans # todo; change the manual setting of top_k to 2 here
        nn_candidates = self.span_cluster_assignment(spans, top_k=1) if self.nn_weight else None

        # # related spans found in domain vocabulary
        kg_candidates = self.span_KG_mapping(spans) if (self.kg_broader_weight or
                                                        self.kg_narrower_weight or
                                                        self.kg_alternatives_weight) else None

        # # PRF ~ spans found in documents that were ranked high based on the initial query
        if self.prf_weight:
            prf_candidates_and_counts = self.pseudo_relevance_feedback(initial_search_results, spans)
        else:
            prf_candidates_and_counts = None

        return spans, nn_candidates, kg_candidates, prf_candidates_and_counts

    def clean_candidate(self, candidate):
        """ Some characters in candidates mess up the querying system"""
        candidate = candidate.replace('"', '').replace("'", '').replace('(', '').replace(')', '')
        candidate = candidate.replace('/', '').replace('\\', '')
        return candidate

    def score_and_select_candidates(self, nn_candidates,
                                    kg_candidates,
                                    prf_candidates_and_counts,
                                    spans, top_k=5):
        """
        Select top X number of candidates; usually ranking is based on IDF values!
        """
        # DONE
        #  - pass IDF_path to the QE class, currently relying on the Clusterer
        #  - tokenize and load the IDF values, currently rely on the Clusterer
        # ToDo;
        #  - select not the top X, but a number of terms based on the initial query length
        #  - pass weights for the different types of candidates
        #  --> (consider moving any IDF related stuff to separate container, currenlty in clustering)
        candidates_and_weights = {}
        if nn_candidates:
            candidates_and_weights["nearest_neighbours"] = {"candidates": nn_candidates,
                                                            "weight": self.nn_weight}
        if kg_candidates:
            candidates_and_weights["kg_broader_terms"] = {"candidates": kg_candidates['broader_terms'],
                                                          "weight": self.kg_broader_weight}
            candidates_and_weights["kg_narrower_terms"] = {"candidates": kg_candidates['narrower_terms'],
                                                           "weight": self.kg_narrower_weight}
            candidates_and_weights["kg_alternative_matches"] = {"candidates": kg_candidates['other_labels'],
                                                                "weight": self.kg_alternatives_weight}

        if prf_candidates_and_counts:
            candidates_and_weights["pseudo_relevant_terms"] = {
                "candidates": [l for l, c in zip(*prf_candidates_and_counts)],
                "counts": [c for l, c in zip(*prf_candidates_and_counts)],
                "weight": self.prf_weight}

        score_counter = Counter()
        qe_insight = {'spans': spans}
        for k, v in candidates_and_weights.items():
            current_candidates_counter = Counter()
            qe_insight[k] = {'weight': v['weight']}
            for idx, candidate in enumerate(v['candidates']):
                url_candidate = parse.quote(candidate)
                response = requests.get(f"{self.cluster}get_idf_weights/{url_candidate}")
                if response:
                    idf_weights = response.json()["idf_weights"]
                    # assuming average IDF weights over the tokens for now
                    weighted_avg_idf_weight = (sum(idf_weights)/len(idf_weights)) * v['weight']
                    if 'counts' in v.keys():
                        # todo multiply with prf_counts? or not?
                        weighted_avg_idf_weight = (sum(idf_weights)/len(idf_weights)) * v['weight'] * v['counts'][idx]
                        candidate = candidate + f", ({str(v['counts'][idx])})"

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
        url_query = parse.quote(initial_query)
        response = requests.get(f"{self.spar}predict_objects/{url_query}").json()
        try:
            return response['prediction']['obj']
        except KeyError as e:
            raise Exception(e, f"{self.spar}predict_objects/{url_query} \n\t gives key 'prediction' not in {response}")

    def span_cluster_assignment(self, spans: List[str], top_k=3) -> List[str]:
        """
        identify nearest neighbours → candidate set 1
        As the input consists of a list of strings, the output would be a list of lists; I think
         -- A list of lists, each nested list holds neighbours of query spans
        we

        :param spans:   List of strings, representing objects that SPaR identified in the query
        :return nearest_neighbours_for_spans:    List of strings, flattened from a list of lists -- where each nested
                                                 list holds neighbours of the input spans
        """
        params = {
            'data': spans,
            'cosine_sim_threshold': 0.7,
            'max_results': top_k
        }
        response = requests.post(f"{self.cluster}get_neighbours/", json=params).json()
        return [nn for nn_list in response['neighbours'] for nn in nn_list]

    def span_KG_mapping(self, spans: List[str]):
        """
        KG expansion → candidate set 2
        """
        query_nodes = [n for n in spans if n.lower() in self.lowercased_nodes]
        expanded_nodes_dict = get_results_for_queries(self.sparql, query_nodes)
        return expanded_nodes_dict

    def pseudo_relevance_feedback(self, initial_search_results, spans):
        """
        PRF - identify which clusters and KG terms occur in initial retrieved docs → candidate set 3
        :input initial_search_results:  This is the `combined_pred` dict that the API's regular query function returns.
        """
        prf_candidates = []
        for idx, score_and_dict in enumerate(initial_search_results):
            score, d = score_and_dict
            # spar_labels = [(str(r), int(c))for r, c in d['label_counters_dict']['SPaR_labels'].most_common()]
            filtered_SPaR_labels = [(str(r), int(c)) for r, c in d['label_counters_dict']['filtered_SPaR_labels'].most_common()]
            # cluster_filtered = [(str(r), int(c)) for r, c in d['label_counters_dict']['cluster_filtered'].most_common()]
            # cluster_neighbours = [(str(r), int(c)) for r, c in d['label_counters_dict']['cluster_neighbours'].most_common()]

            prf_candidates += filtered_SPaR_labels # + spar_labels + cluster_filtered + cluster_neighbours

        prf_counter = Counter()
        for r, c in prf_candidates:
            prf_counter[r] += c
        # todo, return the counts as well here. so i can take them into account later on / check them for tweaking?
        prf_labels = [label for label, count in prf_counter.most_common() if (label != '' and label not in spans)]
        prf_counts = [count for label, count in prf_counter.most_common() if (label != '' and label not in spans)]
        return prf_labels, prf_counts

