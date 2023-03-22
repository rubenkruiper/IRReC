from typing import List, Dict, Any
from collections import Counter
import re


def split_doc_title_and_doc_id(doc_id, title):
    """ For Lee quick solution, should be moved to haystack processing """
    # if this was an XML document, skip all of this
    if " # " in title:
        return title

    # grab BS 1234 2020 type of identifier from filename, seems cleaner solution
    if "--" in doc_id:
        identifier = doc_id.split("--", 1)[0]
    elif "_" in doc_id:
        identifier = doc_id.split("_", 1)[0]
    elif "part" in doc_id:
        identifier = doc_id.split("part", 1)[0]
    else:
        identifier = doc_id.split(".", 1)[0]


    # clean the title if necessary:
    if not title:
        return identifier + " # " + "[Could not grab title]"

    match = None
    if title.startswith("BS") or title.startswith("NA") or title.startswith("+"):
        match = re.match(r"[A-Z +]+(to)?[A-Z ]+([\d\-+:]+)([A-Z ]+[\d\-+:]+)?", title)
    elif title[0].isdigit():
        match = re.match(r'[0-9: +]([\dA-Z\-+:]+)([A-Z ]+[\d\-+:]+)?')
    elif title.startswith("Eurocode"):
        match = re.match(r"Euro([\w :])+([\d\-+:]+)([A-Z ]+[\d\-+:]+)?", title)

    if match:
        end_idx = match.end()
        # less_clean_doc_id = title[:end_idx]
        title = title[end_idx + 1:]

    return identifier + " # " + title


def apply_weights(result_dict: List[Any], weights: Dict[str, int]):
    """
    Applies field specific weights to scores of field specific results.
    """
    possible_fields = ['content', 'doc_title', 'NER_labels', 'filtered_NER_labels',
                       'filtered_NER_labels_domains', 'neighbours', 'bm25_weight']
    results = ((field, predictions) for field, predictions in result_dict.items() if field in possible_fields)
    for field_to_index, predictions in results:
        if field_to_index.startswith("bm25"):
            field_to_index = "bm25"
        field_specific_weight = weights[field_to_index]
        for idx, pred in enumerate(predictions['documents']):
            result_dict[field_to_index]['documents'][idx]['score'] = pred['score'] * field_specific_weight

    return result_dict


def combine_results_from_various_indices(pred_dict: List[Any], weights: Dict[str, int]):
    """
    Our input is a bunch of raw Haystack outputs that are wrapped in a dict as follows:
    {
        'result': {
            {`field_to_index`: {`Haystack output for given field to index`}},
            {`field_to_index`: {`Haystack output for given field to index`}},
            ...
        }
    }

    Each haystack dict itself is in the form:
    {'documents': [`list of documents ranked by score for the field to index`],
     'params': {'Retriever': {'top_k': 10}},
     'query': `format depends on sparse vs dense',
     and some more non-essential information ...
     }

    """
    # First we apply the weights to re-score retrieval result based on the index that returned them
    re_weighted_results = apply_weights(pred_dict, weights)
    # Then we combine results from the different indices;
    # - might be from the same document
    # - may even be the exact same bit of content that was retrieved
    combined_results = {}
    for retrieval_field, retrieved_results in re_weighted_results.items():
        for single_result in retrieved_results['documents']:
            doc_score = single_result['score']
            if not doc_score:
                # skip passages that were retrieved by an index for which the weight is set to 0
                continue

            # doc_id based on filename##pagenr##sectiononpage
            doc_id = single_result['id'].split('##', 1)[0]
            old_doc_title = single_result['meta']['doc_title']
            doc_title = split_doc_title_and_doc_id(doc_id, old_doc_title)

            if doc_title not in combined_results:
                combined_results[doc_title] = RetrievedDoc(doc_title, single_result, doc_score, retrieval_field)
            else:
                combined_results[doc_title].update_content_count_and_score(single_result, doc_score, retrieval_field)

            # update the label stats corresponding to the retrieval_field
            combined_results[doc_title].update_label_stats(single_result, retrieval_field)

    # order passages within each retrieved document by its score
    for doc_obj in combined_results.values():
        doc_obj.combine_contents()
        doc_obj.contents = sorted(doc_obj.contents, key=lambda x: x.score, reverse=True)

    return combined_results


class RetrievedContent:
    def __init__(self, content_dict, field_to_index):
        self.content = content_dict['content']  # May be redundant, but leaving it for now
        self.retrieved_by = [field_to_index]
        self.id = content_dict['id']
        self.score = content_dict['score']

        # meta data
        self.text = content_dict['meta']['text']  # todo this has to be added!
        self.doc_title = content_dict['meta']['doc_title']
        self.split_size = content_dict['meta']['split_size']
        self.split_id = content_dict['meta']['split_id']
        self.SPaR_labels = content_dict['meta']['SPaR_labels']
        self.filtered_SPaR_labels = content_dict['meta']['filtered_SPaR_labels']
        self.cluster_neighbours = content_dict['meta']['cluster_neighbours']
        self.cluster_filtered = content_dict['meta']['cluster_filtered']

    def as_dict(self):
        """ Convert back to dict again... """
        return {
            "content": self.content,
            "retrieved_by": self.retrieved_by,
            "id": self.id,
            "score": self.score,
            "text": self.text,
            "doc_title": self.doc_title,
            "split_size": self.split_size,
            "split_id": self.split_id,
            "SPaR_labels": self.SPaR_labels,
            "filtered_SPaR_labels": self.filtered_SPaR_labels,
            "cluster_neighbours": self.cluster_neighbours,
            "cluster_filtered": self.cluster_filtered
        }


class RetrievedDoc:
    def __init__(self,
                 doc_title,
                 content_dict,
                 score,
                 retrieval_field,
                 label_types=["SPaR_labels", "filtered_SPaR_labels", "cluster_neighbours", "cluster_filtered"]):
        # Aggregated stats for document
        self.doc_title = doc_title
        self.potential_id = self.find_document_identifier(doc_title)
        self.times_retrieved = 1
        self.sum_of_scores = score
        # Aggregated label stats
        self.label_types = label_types
        self.label_counters_dict = {label_types[i]: Counter() for i in range(len(label_types))}
        # All contents retrieved for document
        self.contents = [RetrievedContent(content_dict, retrieval_field)]

    @staticmethod
    def find_document_identifier(title):
        """ Grab the document identifier if we can """
        # TODO this should be in preprocessing / CustomDocument... rather than cleaning up later
        good_starts = ['BS ', "NA ", "+", "P ", "CP ", "D ", "Eurocode ", "ICS "]
        if title.strip().startswith("Lic"):
            title = title.split(",", 1)[1]
        elif not any([title.startswith(g) for g in good_starts]):
            first_match = ''
            best_g = ''
            for g in good_starts:
                if g in title:
                    rest_of_title = title.split(g, 1)[1]
                    if len(rest_of_title) > len(first_match):
                        first_match = rest_of_title
                        best_g = g
            title = best_g + first_match

        match = None
        potential_id = None
        if title:
            if title.startswith("BS") or title.startswith("NA") or title.startswith("+") or \
                    title.startswith("P") or title.startswith("D"):
                match = re.match(r"[A-Z +]+(to)?[A-Z ]+([\d\-+:]+)([A-Z ]+[\d\-+:]+)?", title)
            elif title[0].isdigit():
                match = re.match(r'[0-9: +]([\dA-Z\-+:]+)([A-Z ]+[\d\-+:]+)?')
            elif title.startswith("Eurocode"):
                match = re.match(r"Euro([\w :])+([\d\-+:]+)([A-Z ]+[\d\-+:]+)?", title)

            if match:
                end_idx = match.end()
                potential_id = title[:end_idx]

        return potential_id

    def update_content_count_and_score(self, content_dict, score, retrieval_field):
        # number of contents retreived for this document and overall score
        self.contents.append(RetrievedContent(content_dict, retrieval_field))
        self.times_retrieved += 1
        self.sum_of_scores += score

    def update_label_stats(self, retrieved_dict, retrieval_field):
        # add the label statistics for a specific retrieval_field
        for label_type in self.label_types:
            # I only want to count the labels that contributed to the retrieval of the document/content
            if label_type == retrieval_field:
                for label in retrieved_dict['meta'][label_type].split(", "):
                    self.label_counters_dict[label_type][label] += 1

    def get_label_stats(self, top_k=3):
        """ Need to update this into something more intelligeble """
        for label, counter in self.label_counters_dict.items():
            spans_n_counts = ', '.join([f"{s} ({c})" for s, c in counter.most_common(top_k)])
            print(f" >> {label}: {spans_n_counts}")

    def combine_contents(self):
        """ Make sure all unique contents only occur once, even if they were retrieved multiple times by different fields """
        combined_contents = []
        combined_content_ids = []
        for content_obj in self.contents:
            if content_obj.id in combined_content_ids:
                # same ID retrieve in different ways
                idx = combined_content_ids.index(content_obj.id)
                combined_contents[idx].retrieved_by += content_obj.retrieved_by
                # check for duplicate IDs that have different text somehow
                combined_content = combined_contents[idx]
                if combined_content.text != content_obj.text:
                    print(
                        "ID the same, but contents different:\n{}\n{}".format(combined_content.text, content_obj.text))
                # combine scores (sum for now)
                combined_content.score += content_obj.score
            else:
                combined_contents.append(content_obj)
                combined_content_ids.append(content_obj.id)

        self.contents = combined_contents

    def get_top_contents(self, top_k=3):
        """ Need to update this into something more intelligeble """
        for content_idx, content in enumerate(self.contents[:top_k]):
            print("[{}]{:.2f}: {}".format(str(content_idx + 1), content.score, content.text))

    def as_dict(self):
        return {
            "doc_title": self.doc_title,
            "times_retrieved": self.times_retrieved,
            "sum_of_scores": self.sum_of_scores,
            "label_types": self.label_types,
            "label_counters_dict": self.label_counters_dict,
            "contents": [c.as_dict() for c in self.contents]
        }

