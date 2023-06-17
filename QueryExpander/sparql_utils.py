from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List


# queries I'm not interested in
nodes_one_hop_away = """
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?node
            WHERE {
                ?node ?r ?queryNode . 
                ?queryNode  skos:prefLabel  ?value .
                FILTER (regex (str(?value), "QUERY", "i"))
            }
        """


close_terms = """
    PREFIX bre: <http://purl.org/bre#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    select distinct ?node
    where {
        ?s skos:prefLabel ?node .
        ?s skos:exactMatch|skos:closeMatch ?queryNode .
        ?queryNode skos:prefLabel|skos:altLabel  ?value .
        FILTER (CONTAINS(LCASE(str(?value)), "roof"))
    }
"""

broader_terms = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?node
    WHERE {
        ?s skos:prefLabel ?node .
        ?s ^skos:broader|skos:narrower ?queryNode .
        ?queryNode skos:prefLabel|skos:altLabel  ?value .
        FILTER (CONTAINS(LCASE(str(?value)), "QUERY"))
    }
"""

# queries I'm  interested in
narrower_terms = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?node
    WHERE {
        ?s skos:prefLabel ?node .
        ?s skos:broader|^skos:narrower ?queryNode .
        ?queryNode skos:prefLabel|skos:altLabel  ?value .
        FILTER (CONTAINS(LCASE(str(?value)), "QUERY"))
    }
"""

this_node_terms = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?node
    WHERE {
        ?s skos:prefLabel|skos:altLabel ?node .
        ?s skos:prefLabel|skos:altLabel ?value .
        FILTER (CONTAINS(LCASE(str(?value)), "QUERY"))
    }
"""


def get_results_for_queries(graph_sparql_endpoint: SPARQLWrapper,
                            query_nodes: List[str]):
    all_retrieved_nodes = {}

    all_queries = [broader_terms, this_node_terms, narrower_terms]

    for qtype, q in zip(['broader_terms', 'other_labels', 'narrower_terms'], all_queries):
        all_retrieved_nodes[qtype] = []
        for query_node in query_nodes:
            temp_q = q.replace("QUERY", query_node)
            graph_sparql_endpoint.setQuery(temp_q)
            graph_sparql_endpoint.setReturnFormat(JSON)
            json_output = graph_sparql_endpoint.query().convert()
            all_retrieved_nodes[qtype] += [n['node']['value'] for n in json_output['results']['bindings']]

    return all_retrieved_nodes

