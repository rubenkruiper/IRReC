import glob, json
import numpy as np
import pandas as pd
from collections import Counter
from lib.readers.reader_utils.my_read_utils import *
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


class DatasetCount:

    def __init__(self, file_path, bert_model_name: str = "bert-base-cased"):
        # directory with .txt and .ann files
        self.file_path = file_path

        # containers to store all spans, tags and document names
        self.doc_names = []
        self.sentence_lengths = Counter()
        self.span_dicts = []
        self.all_tags = []

        # Set a pretrained_transformer_tokenizer
        if "uncased" in bert_model_name:
            self.tokenizer = PretrainedTransformerTokenizer(bert_model_name)
        else:
            # Force cased tokenization for SpanBERT
            self.tokenizer = PretrainedTransformerTokenizer(bert_model_name,
                                                            tokenizer_kwargs={"do_lower_case": False})

    def grab_files_from_dir(self):
        """ largely follows dataset reader """
        text_files = sorted(glob.glob(self.file_path + "*.txt"))
        ann_files = sorted(glob.glob(self.file_path + "*.ann"))

        # dataset_as_json = {}

        for text_file, ann_file in zip(text_files, ann_files):
            # Grab the document name, which indicates where the sentence comes from in the original text
            doc_name = text_file.rsplit('/', 1)[1].rsplit('.', 1)[0]
            self.doc_names.append(doc_name)

            with open(text_file, "r") as tf:
                original_sentence = tf.read()

            if original_sentence.startswith("all_figures"):
                # ignore figure references
                print('dataset has a sentence that is a figure reference, that should not happen!!')
                continue

            # Tokenize and compute tags for tokens
            token_list = self.tokenizer.tokenize(original_sentence)
            if len(token_list) < 2:
                # ignore single numbers in tables/lists
                print('dataset has a sentence consisting of a single token, that should not happen!!')
                continue

            # Count spans with BRAT character indices
            span_buffer = annotations_as_dict(ann_file, original_sentence)
            ordered_spans = order_annotations_for_file(span_buffer)
            self.span_dicts.append(ordered_spans)

            # Count tags etc
            bert_indexed_spans = brat_to_PretainedTransformerTokenizer(token_list, ordered_spans)
            tag_list = compute_tags_for_spans(bert_indexed_spans, token_list)
            # ignoring default_tokens [CLS] and [SEP], could simply use tag_list[1:-1]
            tag_list = [t for t in tag_list if t != 'PD-pad']
            self.all_tags += tag_list

            # Sentence lengths
            self.sentence_lengths[len(tag_list)] += 1

        #     dataset_as_json[doc_name] = span_buffer
        #
        # with open('dataset_as_json.json', 'w') as f:
        #     json.dump(dataset_as_json, f, indent=2)

    def print_counts(self):
        """
        self.doc_names: insight in domestic/non-domestic, number of sentences from paragraphs, lists, tables
        self.sentence_lengths: avg sentence length, longest, shortest
        self.span_buffer: number of span types, length of spans per type, average span length,
                          sentence lengths, number of discontiguous, more?
        self.all_tags: number of specific tag types,  more?
        """
        # ToDo - do I want more insight in the doc_names?
        # Division between domestic / non-domestic
        num_domestic = len([d for d in self.doc_names if d.startswith('d')])
        num_non_domestic = len([d for d in self.doc_names if d.startswith('n')])
        print("Number of domestic: {}, and non-domestic: {}".format(num_domestic, num_non_domestic))

        # Sentence stats for dataset
        sent_lengths_list = []
        for num_tokens, counts in self.sentence_lengths.most_common():
            for i in range(counts):
                sent_lengths_list.append(num_tokens)

        print("Mean sentence length (tokens): {}".format(np.mean(sent_lengths_list)))
        print("Standard deviation length (tokens): {}".format(np.std(sent_lengths_list)))
        print("Shortest sentence: {}".format(np.min(sent_lengths_list)))
        print("Longest sentence: {}".format(np.max(sent_lengths_list)))

        # span types, character-based lengths and averages
        span_df = pd.DataFrame({'span_type': [], 'span_length': [], 'discontiguous': []})

        for ann in self.span_dicts:
            for k, v in ann.items():
                length = (v['gap_start'] - v['span_start']) + (v['span_end'] - v['gap_end'])
                disc_bool = v['gap_start'] != v['span_end']
                span_df.loc[len(span_df.index)] = [v['span_type'], length, disc_bool]

        type_counts = span_df.value_counts('span_type')
        total_length_per_type = span_df.groupby('span_type').agg('sum')
        print("Span type counts: \n{}".format(type_counts))
        print("Span type avg lengths: \n{}".format((total_length_per_type.div(type_counts, axis=0))))
        num_discontiguous = span_df.value_counts('discontiguous')
        print("Number of discontiguous spans: {}".format(num_discontiguous[True]))

        # Tag types
        tag_counter = Counter(self.all_tags)
        print("Tag counts: {}".format(tag_counter.most_common()))


if __name__ == "__main__":

    file_paths = ["all_annotated/", "train/", "val/", "test/"]

    # grab annotated files from all_annotated, train, val and test
    for f in file_paths:
        counter = DatasetCount(f)
        print("Counts for {}".format(f))
        counter.grab_files_from_dir()
        counter.print_counts()
        print("\n\n")



