import json, pickle, csv
import regex as re
import numpy as np

from os import path
from collections import Counter
from textblob import TextBlob
from data.ScotReg.parse_utils import *

import spacy
import csv
# import matplotlib.pyplot as plt


class RegulationsInsight:
    """
    ToDo - clean up this code; I initially checked how well chunking and constituency parsing worked on ScotReg.
     Currently the counters are still instantiated, and the (commented) code to use the counters should work fine.
    """

    def __init__(self, tokenizer: str,
                 lemma: bool = False,
                 stem: bool = False):
        self.tokenizer = tokenizer

        if tokenizer == 'spacy':
            self.nlp_split = setup_constituency_parser(const_parse=False)
            # If you'd like to check/compare constituency parsing results, I used benepar for this
            # self.nlp_const = setup_constituency_parser(const_parse=True)

        self.definitions = []
        self.defined_term_count = Counter()
        self.lemma = lemma
        self.stem = stem

        self.vocab_growth = []
        self.word_count = Counter()
        self.sent_len_count = Counter()
        self.noun_chunk_count = Counter()
        self.blob_noun_phrases = Counter()
        self.potential_MWE_count = Counter()
        self.pattern_count = Counter()

    def set_counters(self, cnts):
        self.word_count, self.sent_len_count, self.noun_chunk_count, self.potential_MWE_count, self.pattern_count, \
                self.definitions, self.defined_term_count, self.vocab_growth, self.blob_noun_phrases = cnts

    def print_token_and_sentence_stats(self):
        """
        Prints the statistics for the token and sentence counters
        """
        # Token statistics
        print("\nTOKENS:")
        print("Top tokens: {}".format(self.word_count.most_common(10)))
        print("Total tokens: {}".format(sum([c for k, c in self.word_count.most_common()])))
        print("Vocab size: {}".format(len(self.word_count.keys())))

        # Sentence statistics
        print("\nSentences:")
        print("Total sentences: {}".format(sum([c for r, c in self.sent_len_count.most_common()])))

        counter_list = []
        for num_tokens, counts in self.sent_len_count.most_common():
            for i in range(counts):
                counter_list.append(num_tokens)

        print("Mean sentence length (tokens): {}".format(np.mean(counter_list)))
        print("Standard deviation length (tokens): {}".format(np.std(counter_list)))
        print("Shortest sentence: {}".format(np.min(counter_list)))
        print("Longest sentence: {}".format(np.max(counter_list)))

    def print_noun_and_verb_phrase_stats(self):
        """
        prints the statistics for the definitions, nounchunk, NP and VP counters
        """
        # Definitions
        print("Not (exactly) defined terms: {}".format(len(self.definitions)))
        # print(self.defined_term_count.most_common()[:10])
        # terms_by_length = sorted(self.defined_term_count, key=len, reverse=True)
        # print("Example long terms: {}".format([(t, self.defined_term_count[t]) for t in terms_by_length[:10]]))
        print("Total defined terms: {}".format(len(self.defined_term_count.keys())))
        print("Counts of defined terms: {}".format(sum([c for r, c in self.defined_term_count.most_common()])))

        # # Blob NP statistics
        # print("\nBLOB NPs :")
        # print("Top blob NPS: {}".format(self.blob_noun_phrases.most_common(20)))
        # print("Total blob NPS: {}".format(sum([c for r, c in self.blob_noun_phrases.most_common()])))
        # print("Unique blob NPS: {}".format(len([r for r, c in self.blob_noun_phrases.most_common()])))
        # # noun chunk statistics
        # print("\nNOUN CHUNKS:")
        # print("Top noun chunks: {}".format(self.noun_chunk_count.most_common(20)))
        # print("Total noun chunks: {}".format(sum([c for r, c in self.noun_chunk_count.most_common()])))
        # print("Unique noun chunks: {}".format(len([r for r, c in self.noun_chunk_count.most_common()])))
        # # NP statistics
        # print("\nMWEs - NOUN PHRASES:")
        # print("Top NPs: {}".format(self.potential_MWE_count.most_common(20)))
        # print("Total NPs: {}".format(sum([c for r, c in self.potential_MWE_count.most_common()])))
        # print("Unique MWEs: {}".format(len([r for r, c in self.potential_MWE_count.most_common()])))

        # # NP minus noun chunk
        # blob_nounchunk_diff = [k for k in self.blob_noun_phrases.keys() if k not in self.noun_chunk_count.keys()]
        # blob_NP_diff = [k for k in self.noun_chunk_count.keys() if k not in self.potential_MWE_count.keys()]
        # print("{} Blob NPs that aren't noun_chunks, e.g.: {}".format(len(blob_nounchunk_diff), blob_nounchunk_diff[:20]))
        # print("{} Blob NPs that aren't MWEs, e.g.: {}".format(len(blob_NP_diff), blob_NP_diff[:20]))
        #
        #
        # MWE_not_blob = [k for k in self.potential_MWE_count.keys() if k not in self.blob_noun_phrases.keys()]
        # print("{} MWEs that aren't Blob NPs, e.g.: {}".format(len(MWE_not_blob), MWE_not_blob[:20]))
        # possible_MWEs = [k for k in self.potential_MWE_count.keys() if k not in self.noun_chunk_count.keys()]
        # print("{} MWEs that aren't noun_chunks, e.g.: {}".format(len(possible_MWEs), possible_MWEs[:20]))
        #
        # only_noun_chunks = [k for k in self.noun_chunk_count.keys() if k not in self.potential_MWE_count.keys()]
        # print("{} noun_chunks that aren't MWEs, e.g.: {}".format(len(only_noun_chunks), only_noun_chunks[:20]))


        # # NPs vs definitions
        # blob_NPs_containing_definition = []
        # noun_chunks_containing_definition = []
        # MWEs_containing_definition = []
        # for term in self.definitions:
        #     current_blob_NPs = [k for k in self.blob_noun_phrases.keys() if term in k]
        #     current_noun_chunks = [k for k in self.noun_chunk_count.keys() if term in k]
        #     current_MWEs = [k for k in self.potential_MWE_count.keys() if term in k]
        #     blob_NPs_containing_definition += current_blob_NPs
        #     noun_chunks_containing_definition += current_noun_chunks
        #     MWEs_containing_definition += current_MWEs
        # print("{} blob NPs containing defined terms, e.g.: {}".format(len(blob_NPs_containing_definition), blob_NPs_containing_definition[:20]))
        # print("{} noun_chunks containing defined terms, e.g.: {}".format(len(noun_chunks_containing_definition), noun_chunks_containing_definition[:20]))
        # print("{} Noun Phrases containing defined terms, e.g.: {}".format(len(MWEs_containing_definition), MWEs_containing_definition[:20]))


        # # VP statistics
        # print("\nPatterns - VERB PHRASES:")
        # print("Top VPs: {}".format(self.pattern_count.most_common(10)))
        # print("Total VPs: {}".format(sum([c for r, c in self.pattern_count.most_common()])))

    def unify_text(self, text):
        """
        Stemming and lemmatisation through Textblob (instead of spacy).
        """
        blob = TextBlob(text)
        if not self.lemma and not self.stem:
            # Simply lowercase string
            text_ = ' '.join([w.lower() for w in blob.words])
        elif self.lemma:
            text_ = ' '.join([t.lemmatize().lower() for t in blob.words])
        elif self.stem:
            text_ = ' '.join([t.stem().lower() for t in blob.words])
        return text_

    def grab_defined_terms_and_clean_sent(self, sentence_string):
        """
        During scraping (code not provided in this repo) I tag defined terms, as well as:
         - figures, text from table cells, list items
         - explicit quotes and notes in the text
        """

        defined_terms = re.findall('#DEFINED#(.*?)#TERM#', sentence_string)
        for defined_term in defined_terms:
            sentence_string = re.sub('#DEFINED#{}#TERM#'.format(defined_term), defined_term, sentence_string)
            unified_term = self.unify_text(defined_term)

            if unified_term in self.definitions:
                # check which definitions were found in text
                self.definitions.remove(unified_term)

            self.defined_term_count[unified_term] += 1

        if "#DEFINED#" in sentence_string:
            # sometimes a the regex sub pattern is thrown off by parentheses, a URL, or whatever
            still_has_to_be_removed = re.findall('#DEFINED#(.*?)#TERM#', sentence_string)
            for d in still_has_to_be_removed:
                p1, p2 = sentence_string.split(d, 1)
                sentence_string = p1[:-9] + d + p2[6:]

        # remove QUOTE, NOTE, and Figure indicators
        if "#QUOTE" in sentence_string:
            sentence_string = sentence_string[7:-11]
        elif "#NOTE" in sentence_string:
            sentence_string = sentence_string[6:-10]
        elif "#Figure" in sentence_string:
            sentence_string = re.sub(r"#Figure\d{1,2}#", '', sentence_string)
            sentence_string = re.sub(r"#EndFigure\d{1,2}", '', sentence_string)
            if sentence_string.endswith('#'):
                sentence_string = sentence_string[:-1]

        return sentence_string

    def grab_text_from_lists_and_tables(self, list_of_lists):

        def grab_nested_item(some_list):
            plain_text_ = []
            for i in some_list:
                if type(i) == list:
                    plain_text_ = grab_nested_item(i)
                elif type(i) == str:
                    i = re.sub("#Li[0-9_]*#", '', i)
                    if len(i) > 4:
                        if i.startswith("['"):
                            # handle lists that have been converted to a string for some reason
                            i_as_list = eval(i)
                            plain_text_ = grab_nested_item(i_as_list)
                            continue
                    plain_text_.append(self.grab_defined_terms_and_clean_sent(re.sub("#Li[0-9_]*#", '', i)))
            return plain_text_

        plain_text = []
        for item in list_of_lists:
            if type(item) == str:
                if not item.startswith("#EndTable") and not item.startswith("#Table"):
                    plain_text.append(self.grab_defined_terms_and_clean_sent(re.sub("#Li[0-9_]*#", '', item)))
            elif type(item) == list:
                plain_text += grab_nested_item(item)

        return plain_text

    def parse_using_textblob(self, input_text):
        blob = TextBlob(input_text)

        for sent in blob.sentences:
            # ToDo - currently figures are saved in a separate folder, and their location in the text is marked
            #  this results in non-sentence entries in ScotReg
            # if sent.raw.startswith("all_figures"):
            #     # ignore figure references
            #     continue

            tokens = [w for w in sent.words]
            # ToDo - sometimes a table contains cells that act as a numbered list, which is completely useless...
            # if len(tokens) < 2:
            #     # ignore single numbers in tables/lists
            #     continue

            # count sentence lengths
            # if len(tokens) in self.sent_len_count:
            self.sent_len_count[len(tokens)] += 1
            # else:
            #     self.sent_len_count[len(tokens)] = 1

            # count words (note that this excludes punctuation)
            for t in tokens:
                if str(t) in self.word_count:
                    self.word_count[str(t)] += 1
                else:
                    self.word_count[str(t)] = 1
                    self.vocab_growth.append(sum([c for k, c in self.word_count.most_common()]))
            # We can also use TextBlob for NPs, so count them here to compare
            for blob_np in sent.noun_phrases:
                self.blob_noun_phrases[str(blob_np)] += 1
        return [str(s) for s in blob.sentences]

    def parse_using_spacy(self, input_text):
        """ simply count words and sentences """
        doc = self.nlp_split(input_text)
        sent_list = [s.text for s in doc.sents]

        for sent in doc.sents:

            if sent.text.startswith("all_figures"):
                # ignore figure references
                continue

            tokens = get_tokens(sent)
            # if len(tokens) < 2:
            #     # ignore single numbers in tables/lists
            #     continue

            # count sentence lengths
            if len(tokens) in self.sent_len_count:
                self.sent_len_count[len(tokens)] += 1
            else:
                self.sent_len_count[len(tokens)] = 1
            # count words
            for t in tokens:
                if t.text in self.word_count:
                    self.word_count[t.text] += 1
                else:
                    self.word_count[t.text] = 1
                    self.vocab_growth.append(sum([c for k, c in self.word_count.most_common()]))

        return sent_list

    def parse_single_paragraph_or_sentence(self, input_text):
        """ constituency parse sentences; also count NPs, tokens, sentences, etc. """
        if self.tokenizer == 'textblob':
            sent_list = self.parse_using_textblob(input_text)
        elif self.tokenizer == 'spacy':
            # currently not constituency parsing
            sent_list = self.parse_using_spacy(input_text)

        # # run spacy for NP and const parsing
        # spacy_doc_per_sentence = [self.nlp_const(s) for s in sent_list]
        # sentences = []
        # for doc in spacy_doc_per_sentence:
        #     sentences += list(doc.sents)
        #
        # for sent in sentences:
        #     if 'parser' in self.nlp_const.pipe_names:
        #         # NounChunks does require the spacy dependency parser
        #         # Count all NPs found by spacy as noun_chunks
        #         spacy_noun_chunks = get_spacy_noun_chunks(sent)
        #         for noun_phrase in spacy_noun_chunks:
        #             if noun_phrase in self.noun_chunk_count:
        #                 self.noun_chunk_count[noun_phrase] += 1
        #             else:
        #                 self.noun_chunk_count[noun_phrase] = 1
        #
        #     if 'benepar' in self.nlp_const.pipe_names:
        #         # Count all NPs in the constituency tree
        #         constituent_nps = grab_subtrees_of_type(sent, 'NP')
        #         for noun_phrase_long in constituent_nps:
        #             # some unification of NPs starting (the X) and (a Y)
        #             if noun_phrase_long.startswith('the '):
        #                 noun_phrase = noun_phrase_long.split('the ', 1)[1]
        #             elif noun_phrase_long.startswith('a '):
        #                 noun_phrase = noun_phrase_long.split('a ', 1)[1]
        #             elif noun_phrase_long.startswith('The '):
        #                 noun_phrase = noun_phrase_long.split('The ', 1)[1]
        #             elif noun_phrase_long.startswith('A '):
        #                 noun_phrase = noun_phrase_long.split('A ', 1)[1]
        #             else:
        #                 noun_phrase = noun_phrase_long
        #
        #             if noun_phrase in self.potential_MWE_count:
        #                 self.potential_MWE_count[noun_phrase] += 1
        #             else:
        #                 self.potential_MWE_count[noun_phrase] = 1
        #
        #         # count all VPs (minus the subject/object NP)
        #         constituent_vps = grab_subtrees_of_type(sent, 'VP')
        #         for verb_phrase in constituent_vps:
        #             if verb_phrase in self.pattern_count:
        #                 self.pattern_count[verb_phrase] += 1
        #             else:
        #                 self.pattern_count[verb_phrase] = 1


    def grab_definitions(self, file_paths, data={}):
        for fp in file_paths:

            if "/" in fp:
                f_name = fp.rsplit('/', 1)[1].rsplit('.', 1)[0]
            else:
                f_name = fp.rsplit('.', 1)[0]

            with open(fp) as f:
                data[f_name] = json.load(f)

        # first grab all definitions
        for file_name, sections in data.items():
            for subtitle in sections.keys():
                if subtitle.startswith('Appendix A.'):
                    subsubtitle = [x for x in sections[subtitle].keys() if x != 'url'][0]
                    definitions = [k for k in sections[subtitle][subsubtitle]['definitions'].keys() if k != '']
                    for d in definitions:
                        if d not in self.definitions:
                            self.definitions.append(self.unify_text(d))
        return data


    def count(self, file_paths):
        data = {}
        for fp in file_paths:
            print("Counting for file: {}".format(fp))

            if "/" in fp:
                f_name = fp.rsplit('/', 1)[1].rsplit('.', 1)[0]
            else:
                f_name = fp.rsplit('.', 1)[0]

            with open(fp) as f:
                data[f_name] = json.load(f)

        # first grab all definitions
        data = self.grab_definitions(file_paths, data)

        print("Total definitions: {}".format(len(list(set(self.definitions)))))

        for file_name, sections in data.items():
            for subtitle in sections.keys():
                if not subtitle.startswith('Appendix A.'):
                    for subsubtitle, subsubsection in sections[subtitle].items():
                        for subsubsubtitle, text_list in subsubsection.items():
                            if subsubsubtitle != 'url':
                                for item in text_list:
                                    if type(item) == list:
                                        plain_text_list = self.grab_text_from_lists_and_tables(item)
                                        for clean_sent in plain_text_list:
                                            self.parse_single_paragraph_or_sentence(clean_sent)
                                    elif type(item) == str:
                                        if "#" in item:
                                            clean_sent = self.grab_defined_terms_and_clean_sent(item)
                                            self.parse_single_paragraph_or_sentence(clean_sent)
                                        else:
                                            self.parse_single_paragraph_or_sentence(item)
                                    elif not item:
                                        # occasionally None
                                        pass


def count_and_print_insights(tokenizer, file_paths, pkl_file="counts.pkl"):

    reg_cntr = RegulationsInsight(tokenizer, lemma=True)
    counters = [reg_cntr.word_count, reg_cntr.sent_len_count, reg_cntr.noun_chunk_count,
                reg_cntr.potential_MWE_count, reg_cntr.pattern_count, reg_cntr.definitions,
                reg_cntr.defined_term_count, reg_cntr.vocab_growth, reg_cntr.blob_noun_phrases]

    if path.isfile(pkl_file):
        with open(pkl_file, 'rb') as f:
            counters = pickle.load(f)
        reg_cntr.set_counters(counters)
    else:
        reg_cntr.count(file_paths)
        with open(pkl_file, 'wb') as f:
            pickle.dump(counters, f, protocol=pickle.HIGHEST_PROTOCOL)

    reg_cntr.grab_definitions(file_paths)
    reg_cntr.print_token_and_sentence_stats()
    reg_cntr.print_noun_and_verb_phrase_stats()

    # Plot vocab growth ~ saving counts for interpolation and comparison
    # x_numtokens =[]
    # y_vocabsize =[]
    # for y, x in enumerate(reg_cntr.vocab_growth):
    #     x_numtokens.append(x)
    #     y_vocabsize.append(y)
    #
    # with open('../notebooks/scotreg_vocab_growth.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(zip(x_numtokens, y_vocabsize))
    #
    # plt.plot(x_numtokens, y_vocabsize)
    # plt.xlabel("Number of Tokens (N)")
    # plt.ylabel("Vocabulary size (V)")
    # plt.axvline(x=152380, color='red', linestyle='--')      # 152.380 = num_tokens domestic regulations
    # plt.show()


if __name__ == "__main__":
    domestic = "domestic_standards.json"
    non_domestic = "non-domestic_standards.json"
    file_paths = [domestic, non_domestic]
    pkl_file = "counts.pkl"
    pkl_files = ["domestic_counts.pkl", "non_domestic_counts.pkl"]

    # Choice of 'spacy' and 'textblob' ~  Using textblob because spacy does worse in terms of sentence splitting.
    tokenizer = 'textblob'
    # for counting word-level tokens, note that TextBlob doesnt count punctuation

    # counts per file:
    # for fp, pkl in zip(file_paths, pkl_files):
    #     print("#################################\n# DOMESTIC VS NON DOMESTIC \n################################")
    #     count_and_print_insights(tokenizer, [fp], pkl)

    print("#################################\n# DOMESTIC & NON-DOMESTIC COUNTS \n################################")
    count_and_print_insights(tokenizer, file_paths)


