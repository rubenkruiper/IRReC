import json, pickle, os
import numpy as np
import pandas as pd
from collections import Counter
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from spar_lib.readers.reader_utils.my_read_utils import *


class PredictionInsight():
    """ Grab objects from an predictions output_file """

    def __init__(self,
                 bert_model_name: str ="bert-base-cased"):   # todo  "whaleloops/phrase-bert"
        self.predictions_input = ''

        if bert_model_name is not None:
            self.tokenizer = PretrainedTransformerTokenizer(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name

        self.discontiguous_obj_count = 0
        self.discontiguous_act_count = 0
        self.all_spans = {"objects": [],
                          "actions": [],
                          "functional": [],
                          "discourse": []}

        self.document_sent_count = Counter()
        self.sent_tokenlen_list = []
        # span types, character-based lengths and averages
        self.span_df = pd.DataFrame({'span_type': [], 'span_length': []})

        self.tag_count = Counter()

    def print_stats(self, count_spans):
        """
        Prints the collected counts
        """
        print("\nSentences:")

        print(self.document_sent_count.most_common())
        print("Total sentences: {}".format(len(self.sent_tokenlen_list)))

        print("Mean sentence length (tokens): {}".format(np.mean(self.sent_tokenlen_list)))
        print("Standard deviation length (tokens): {}".format(np.std(self.sent_tokenlen_list)))
        print("Shortest sentence: {}".format(np.min(self.sent_tokenlen_list)))
        print("Longest sentence: {}".format(np.max(self.sent_tokenlen_list)))

        if count_spans:
            type_counts = self.span_df.value_counts('span_type')
            total_length_per_type =  self.span_df.groupby('span_type').agg('sum')
            print("Span type counts: \n{}".format(type_counts))
            print("Span type avg lengths: \n{}".format((total_length_per_type.div(type_counts, axis=0))))

        # Tag types
        tag_counter = Counter(self.tag_count)
        print("Tag counts: {}".format(tag_counter.most_common()))

        print("Found {} discontiguous spans of type 'object'".format(self.discontiguous_obj_count))
        print("Found {} discontiguous spans of type 'action'".format(self.discontiguous_act_count))

    def count_doc_id(self, doc_id):
        if doc_id.startswith('d'):
            self.document_sent_count['domestic'] += 1
        else:
            self.document_sent_count['non-domestic'] += 1

    def count_sent_length_in_tokens(self, token_list):
        # Don't count CLS and SEP
        self.sent_tokenlen_list.append(len(token_list[1:-1]))

    def count_tag_types(self, tag_list):
        # Don't count CLS and SEP
        for tag in tag_list[1:-1]:
            self.tag_count[tag] += 1

    def count_span_types(self):
        for span_type in self.all_spans.keys():
            for span in self.all_spans[span_type]:
                length = sum([len(token) for token in span])
                self.span_df.loc[len(self.span_df.index)] = [span_type, length]

    def read_and_count(self, predictions_fp):
        """  """
        # read predictions from output file
        with open(predictions_fp, 'r') as f:
            predictions_list = [json.loads(jsonline) for jsonline in f.readlines()]

        for prediction in predictions_list:
            mask = prediction['mask']
            tag_list = prediction['tags']
            token_list = prediction['words']
            sentence = prediction["sentence"]

            # if sentence.startswith("all_figures"):
            #     # ignore figure references
            #     continue

            # if len(token_list) < 2:
            #     # ignore single numbers in tables/lists
            #     continue

            self.count_doc_id(prediction["doc_id"])
            self.count_sent_length_in_tokens(token_list)
            self.count_tag_types(tag_list)

            # not sure if this is necessary; doesn't seem to be
            predicted_tags = [t for m, t in zip(mask, tag_list) if m]
            mwes, discontiguous_count = self.get_mwes(token_list, predicted_tags, 'obj')
            self.discontiguous_obj_count += discontiguous_count
            self.all_spans["objects"] += mwes

            mwes, discontiguous_count = self.get_mwes(token_list, predicted_tags, 'act')
            self.discontiguous_act_count += discontiguous_count
            self.all_spans["actions"] += mwes

            dis, _ = self.get_mwes(token_list, predicted_tags, 'dis')
            self.all_spans["discourse"] += dis
            func, _ = self.get_mwes(token_list, predicted_tags, 'func')
            self.all_spans["functional"] += func

    def set_counters(self, cnts):
        self.document_sent_count, self.sent_tokenlen_list, self.span_df, self.tag_count, \
                self.all_spans, self.discontiguous_obj_count, self.discontiguous_act_count = cnts

    def grab_counts(self, predictions_fp, count_pickle='predictions/counts.pkl', count_spans=False):
        """
        Count and print the number of different span types and lengths.
        """
        counters = [self.document_sent_count,
                    self.sent_tokenlen_list,
                    self.span_df,
                    self.tag_count,
                    self.all_spans,
                    self.discontiguous_obj_count,
                    self.discontiguous_act_count]

        if os.path.isfile(count_pickle):
            with open(count_pickle, 'rb') as f:
                counters = pickle.load(f)
            self.set_counters(counters)
        else:
            self.read_and_count(predictions_fp)
            if count_spans:
                self.count_span_types()
            with open(count_pickle, 'wb') as f:
                pickle.dump(counters, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.print_stats(count_spans)

        return self.all_spans

    def get_mwes(self, word_list, tag_list, mwe_type):
        """
        Collects the MWEs found by a trained tagger, handles discontiguous spans.
        """

        mwes = []
        current_mwe = []
        previous_head = []
        just_removed = []
        number_of_discontiguous_mwes = 0

        for idx, t in enumerate(tag_list):
            t_head, t_type = t.split('-')
            if t_type != mwe_type:
                # We're only interested in collecting a specific type of MWE for now, e.g., objects / actions
                if current_mwe != []:
                    # store mwe
                    mwes.append(current_mwe)
                    # ToDo - I do not always want to store current_mwe into previous head
                    if just_removed == []:
                        # als we net een BD hebben gehad, dan moet previous_head niet veranderen...
                        previous_head = current_mwe
                current_mwe = []
                continue

            if t_head == 'BH':
                just_removed = []

                if current_mwe == []:
                    # Start collecting a new mwe
                    current_mwe.append(word_list[idx])
                else:
                    # Store old mwe and start collecting a new mwe
                    mwes.append(current_mwe)
                    previous_head = current_mwe
                    current_mwe = [word_list[idx]]
            elif t_head == 'IH':
                # Continue collecting the same object
                current_mwe.append(word_list[idx])
            elif t_head == 'BD':
                number_of_discontiguous_mwes += 1
                # Remove singleton previous_head from mwes once
                if just_removed != previous_head and previous_head in mwes:
                    mwes.reverse()
                    mwes.remove(previous_head)
                    mwes.reverse()
                    just_removed = previous_head

                # Append token.text to previous_mwe
                current_mwe = previous_head + [word_list[idx]]
            elif t_head == 'ID':
                current_mwe.append(word_list[idx])

        return mwes, number_of_discontiguous_mwes

    def mwe_list_to_string(self, mwe_list):
        text = ''
        for w in mwe_list:
            if w.startswith('##'):
                text += w[2:]
            else:
                text += " " + w

        text = text[1:]
        if text.startswith('the ') or text.startswith('The '):
            text = text[4:]
        elif text.startswith('a ') or text.startswith('A '):
            text = text[2:]
        elif text.startswith('an ') or text.startswith('An '):
            text = text[3:]
        return text

    def count_mwes(self, all_mwes):
        counter_list = []
        for k in all_mwes:
            counter = Counter()
            for mwe in all_mwes[k]:
                counter[self.mwe_list_to_string(mwe)] += 1
            counter_list.append(counter)
        return counter_list


def grab_definitions(file_paths):
    data = {}
    for fp in file_paths:
        f_name = fp.rsplit('/', 1)[1].rsplit('.', 1)[0]
        with open(fp) as f:
            data[f_name] = json.load(f)

    definitions = []
    for file_name, sections in data.items():
        for subtitle in sections.keys():
            if subtitle.startswith('Appendix A.'):
                subsubtitle = [x for x in sections[subtitle].keys() if x != 'url'][0]
                doc_definitions = [k for k in sections[subtitle][subsubtitle]['definitions'].keys() if k != '']
                for d in doc_definitions:
                    if d not in definitions:
                        definitions.append(d)
    return definitions


def compare_predictions_against_defined_terms(file_paths, mwe_counter_lists):
    definitions = grab_definitions(file_paths)

    for counter in mwe_counter_lists:
        print("Top 20 counts: \n{}".format(counter.most_common(20)))

    objects_lower = [x.lower().strip() for x in mwe_counter_lists[0]]
    # actions_lower = [x.lower() for x in mwe_counter_lists[1]]
    defined_not_found = [d for d in definitions if d.lower().strip() not in objects_lower]

    for idx, d in enumerate(defined_not_found):
        if d[0] in ["'", '"'] and d[-1] in ["'", '"']:
            defined_not_found[idx] = d[1:-1]

    defined_part_of = []
    defined_actions = []
    for d in defined_not_found:
        overlapping_objects = [x for x in objects_lower if d.lower() in x]
        if len(overlapping_objects) > 0:
            print("{} \npart of \n{}\n".format(d, overlapping_objects))
            defined_part_of.append(d)
            continue

        for o in objects_lower:
            if '-' in d:
                # tokenization is different with -
                d_parts = d.split('-', 1)
                d_new = ' - '.join(d_parts)
                if d_new.lower() in o:
                    defined_part_of.append(d)
                    break

    defined_not_found = [x for x in defined_not_found if x not in defined_part_of]
    # Lazy solution right now is to add a break-point here to inspect the identified objects
    [print(dn) for dn in defined_not_found]

if __name__ == "__main__":
    # file_paths to grab the definitions
    file_paths = ['../i-ReC/classifier_data/scottish/domestic_standards.json',
                  '../i-ReC/classifier_data/scottish/non-domestic_standards.json']

    # Counters are stored in count_pickle
    my_pred_obj = PredictionInsight()
    mwe_dict = my_pred_obj.grab_counts('predictions/all_sentence_predictions.json',
                                       count_pickle='predictions/counts.pkl',
                                       count_spans=True)

    # mwe_dict = my_pred_obj.collect_mwes('predictions/debug_output.json')
    mwe_counter_lists = my_pred_obj.count_mwes(mwe_dict)

    # check which defined terms we find
    compare_predictions_against_defined_terms(file_paths, mwe_counter_lists)
