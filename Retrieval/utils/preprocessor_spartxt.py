import re
import nltk
import logging
import requests
import urllib.parse
from tqdm import tqdm
from copy import deepcopy
from typing import List, Optional, Generator, Set, Union
from textblob import TextBlob
from functools import partial, reduce
from itertools import chain
from collections import Counter

from utils import cleaning_utils
from haystack.nodes.preprocessor import BasePreProcessor

logger = logging.getLogger(__name__)


iso639_to_nltk = {
    "ru": "russian",
    "sl": "slovene",
    "es": "spanish",
    "sv": "swedish",
    "tr": "turkish",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "it": "italian",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
}


class SparPreProcessor(BasePreProcessor):
    def __init__(
        self, ner_url: str = 'http://0.0.0.0:8501/',
        regex_filter: str = 'yes',
        clean_whitespace: bool = True,
        clean_header_footer: bool = False,
        clean_empty_lines: bool = True,
        output_type: str = "text",
        split_by: str = "word",
        split_length: int = 100,
        split_overlap: int = None,
        language: str = "en",
        split_respect_sentence_boundary: Optional[bool] = True
    ):
        """
        This is a Haystack BasePreProcessor that first calls SPaR.txt, and then prepares sentences for indexing.
         1 Provide url for the SPaR.txt API during init
         2 Feed each sentence to SPaR.txt during splitting of the preprocessor
         3 Handle sentences that are too long for BERT models (512 tokens)
         4 Reformat SPaR.txt outputs for Retrieval, relies for some part on SPaR_utils.py

        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                     or similar.
        :param clean_whitespace: Strip whitespaces before or after each line in the text.
        :param clean_empty_lines: Remove more than two empty lines in the text.
        :param output_type: Type of output to keep. Can be "text", "objects", or "text_and_objects". Set to None to disable splitting.
        :param split_by: Unit for splitting the document. Can be "word", "sentence", or "passage". Set to None to disable splitting.
        :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document. For instance, if n -> 10 & split_by ->
                           "sentence", then each output document will have 10 sentences.
        :param split_overlap: Word overlap between two adjacent documents after a split.
                              Setting this to a positive number essentially enables the sliding window approach.
                              For example, if split_by -> `word`,
                              split_length -> 5 & split_overlap -> 2, then the splits would be like:
                              [w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12].
                              Set the value to 0 to ensure there is no overlap among the documents after splitting.
        :param split_respect_sentence_boundary: Whether to split in partial sentences if split_by -> `word`. If set
                                                to True, the individual split will always have complete sentences &
                                                the number of words will be <= split_length.
        :param language: The language used by "nltk.tokenize.sent_tokenize" in iso639 format. Available options: "en", "es", "de", "fr" & many more.
        """
        self.ner_url = ner_url

        # Note: respecting sentence boundaries is statically set to True, to avoid degrading SPaRtxt results.
        split_respect_sentence_boundary = True

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # Regex based initial cleaning of SPaR.txt outputs
        self.regex_filter = cleaning_utils.RegexFilter() if regex_filter in ['yes', 'y', 'Y', 'Yes'] else None

        self.clean_whitespace = clean_whitespace
        self.clean_header_footer = clean_header_footer
        self.clean_empty_lines = clean_empty_lines
        self.output_type = output_type
        self.split_by = split_by
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_respect_sentence_boundary = split_respect_sentence_boundary
        self.language = iso639_to_nltk.get(language, language)
        self.print_log: Set[str] = set()

    def cleaning_helper(self, to_be_cleaned: List[str], minimum_count: int = 2):
        """
        Helper function to call the basic filtering steps outlined in the cleaning utilities script.
        """
        _, regex_cleaned = self.regex_filter.run_filter(
            to_be_cleaned)  # _ would be the list of terms removed by our regex filters
        basic_cleaned = cleaning_utils.custom_cleaning_rules(regex_cleaned)
        determiners_removed = [cleaning_utils.remove_determiners(t) for t in basic_cleaned]
        cleaned_terms = [t for t in determiners_removed if t]
        cleaned_counter = Counter(cleaned_terms)

        # We only want terms to occur at least minimum_count times in a corpus, e.g., 2 times:
        cleaned_terms = [t for t in cleaned_terms if cleaned_counter[t] >= minimum_count]
        return cleaned_terms

    def predict_objects(self, sentence: str):
        # todo;
        #  - would I want to have multiple predictors ready to go?
        #  - speed up NER processing by providing a (batch_size) number of sentences in one go
        response = requests.post(f"{self.ner_url}predict_objects/",
                                 json={"sentence": sentence}).json()
        try:
            response["prediction"]
            return response
        except KeyError as e:
            # raise Exception(e, f"{self.ner_url}predict_objects/{text_to_process} \n\t gives key 'prediction' " +
            #                 f"not in {response}")
            # todo; figure out which inputs cause this, and filter them out before
            return None

    def process(
        self,
        content_dicts: Union[dict, List[dict]],
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        output_type: Optional[str] = None,
        split_by: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
    ) -> List[dict]:
        """
        Perform document cleaning and splitting. Can take a single document or a list of documents as input and returns a list of documents.
        """

        kwargs = {
            "clean_whitespace": clean_whitespace,
            "clean_header_footer": clean_header_footer,
            "clean_empty_lines": clean_empty_lines,
            "output_type": output_type,
            "split_by": split_by,
            "split_length": split_length,
            "split_overlap": split_overlap,
            "split_respect_sentence_boundary": split_respect_sentence_boundary
        }

        ret = []

        if type(content_dicts) == dict:
            ret = self._process_single(
                content_dict=content_dicts,
                **kwargs                #type: ignore
        )
        elif type(content_dicts) == list:
            ret = self._process_batch(
                documents=list(content_dicts),
                **kwargs
            )

        else:
            raise Exception("documents provided to PreProcessor.prepreprocess() is not of type list nor Document")

        return ret

    def _process_single(
        self,
        content_dict,
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        output_type: Optional[str] = None,
        split_by: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
    ) -> List[dict]:
        """
        Main processing function, this is the one that will have to call the SPaRtxt predictor
        """

        if clean_whitespace is None:
            clean_whitespace = self.clean_whitespace
        if clean_header_footer is None:
            clean_header_footer = self.clean_header_footer
        if clean_empty_lines is None:
            clean_empty_lines = self.clean_empty_lines
        if output_type is None:
            output_type = self.output_type
        if split_by is None:
            split_by = self.split_by
        if split_length is None:
            split_length = self.split_length
        if split_overlap is None:
            split_overlap = self.split_overlap
        if split_respect_sentence_boundary is None:
            split_respect_sentence_boundary = self.split_respect_sentence_boundary

        cleaned_document = self.clean(
            document=content_dict,
            clean_whitespace=clean_whitespace,
            clean_header_footer=clean_header_footer,
            clean_empty_lines=clean_empty_lines,
        )

        split_documents = self.split(
            content_dict=cleaned_document,
            output_type=output_type,
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_respect_sentence_boundary=split_respect_sentence_boundary,
        )

        return split_documents

    def _process_batch(
        self,
        documents: List[dict],
        **kwargs
    ) -> List[dict]:
        nested_docs = [self._process_single(d, **kwargs) for d in tqdm(documents, unit="docs")]
        return [d for x in nested_docs for d in x]

    def clean(
        self,
        document: dict,
        clean_whitespace: bool,
        clean_header_footer: bool,
        clean_empty_lines: bool,
    ) -> dict:
        """
        Perform document cleaning on a single document and return a single document. This method will deal with whitespaces, headers, footers
        and empty lines. Its exact functionality is defined by the parameters passed into PreProcessor.__init__().
        """
        text = document["content"]
        if clean_header_footer:
            text = self._find_and_remove_header_footer(
                text, n_chars=300, n_first_pages_to_ignore=1, n_last_pages_to_ignore=1
            )

        if clean_whitespace:
            lines = text.splitlines()

            cleaned_lines = []
            for line in lines:
                line = line.strip()
                cleaned_lines.append(line)
            text = "\n".join(cleaned_lines)

        if clean_empty_lines:
            text = re.sub(r"\n\n+", "\n\n", text)

        document["content"] = text
        return document

    def split(
        self,
        content_dict: dict,
        output_type: str,
        split_by: str,
        split_length: int,
        split_overlap: int,
        split_respect_sentence_boundary: bool = True,
    ) -> List[dict]:
        """
        Perform document splitting on a single page of a document. This method can split on different units, at
        different lengths, with different strides. It can also respect sentence boundaries. Its exact functionality is
        defined by the parameters passed into PreProcessor.__init__(). Takes a single document as input and returns a
        list of documents.
        """

        if not split_by:
            return [content_dict]

        if not split_length:
            raise Exception("split_length needs be set when using split_by.")

        if not split_respect_sentence_boundary:
            raise NotImplementedError("'split_respect_sentence_boundary=False' has not been implemented.")

        # remove consecutive spaces, but leave newlines
        text = ' '.join([x for x in content_dict["content"].split(' ') if x != ''])

        if split_respect_sentence_boundary:
            # ensuring the split always happens at a sentence-boundary
            sentences = []
            for part in text.split('\n'):
                sentences += [str(s) for s in TextBlob(part).sentences]
            cumulative_len = 0

            # splits to store text
            list_splits = []
            current_slice: List[str] = []

            # splits to store tags
            list_tag_splits = []
            current_tags_slice: List[str] = []

            for sen in sentences:
                prediction_dict = self.predict_objects(sen)
                if not prediction_dict:
                    continue
                pred_labels = prediction_dict["prediction"]
                current_token_count = prediction_dict["num_input_tokens"]
                span_token_count = prediction_dict["num_output_tokens"]

                # ways of combining text with labels #todo deprecate
                # -- Prep context todo; consider removing the choice of output type
                if output_type == 'text':
                    pass
                elif output_type == 'objects':
                    sen = ' '.join(pred_labels['obj'])
                elif output_type == "text_and_objects":
                    # append objects to the sentence text
                    sen += ' ' + ' '.join(pred_labels['obj'])

                # -- Determine sequence length todo; consider removing the choice of output type
                if split_by == "word":
                    # ignoring punctuation
                    sequence_len = len(sen.split(" "))
                elif split_by == 'token':
                    if output_type == 'text':
                        sequence_len = current_token_count
                    elif output_type == 'objects':
                        sequence_len = span_token_count
                    elif output_type == "text_and_objects":
                        sequence_len = current_token_count + span_token_count

                # -- Truncate sequences that are by default longer than split_length
                if sequence_len > split_length:
                    long_sentence_message = f"One or more sentences found with word count higher than the split length."
                    if long_sentence_message not in self.print_log:
                        self.print_log.add(long_sentence_message)
                        logger.warning(long_sentence_message)

                    sen = sen[:300]  # todo; better way to truncate ? > assuming split_length aims
                    # continue # skip sentence?

                # -- Determine if the current split is appended to previous, or starts a new split
                if cumulative_len + sequence_len > split_length:
                    # append what we have so far to splits, and start new split
                    list_splits.append(current_slice)
                    list_tag_splits.append(current_tags_slice)
                    if not split_overlap:
                        current_slice = []
                        current_tags_slice = []
                        cumulative_len = 0
                    else:
                        # Enable split_stride with split_by='word' while re specting sentence boundaries.
                        # NOTE: we set split_overlap to 0 words overlap by default, rather than None
                        overlap = []
                        w_count = 0
                        for s in current_slice[::-1]:
                            sen_len = len(s.split(" "))
                            if w_count < split_overlap:
                                overlap.append(s)
                                w_count += sen_len
                            else:
                                break
                        current_slice = list(reversed(overlap))
                        cumulative_len = w_count

                current_slice.append(sen)
                current_tags_slice.append(pred_labels['obj'])
                cumulative_len += sequence_len

            if current_slice:
                list_splits.append(current_slice)
                list_tag_splits.append(current_tags_slice)

            text_splits = []
            tag_splits = []
            for sl, tl in zip(list_splits, list_tag_splits):
                txt = ' '.join(sl)
                # print(f"[TL]: {tl}")
                # print(f"[tags]: {[t for tl_ in tl for t in tl_]}")
                if tl:
                    tags = [t for tl_ in tl for t in tl_]
                else:
                    pass

                if len(txt) > 0:
                    text_splits.append(txt)
                    tag_splits.append(tags)
        else:
            raise NotImplementedError("PreProcessor only supports 'passage', 'sentence' or 'word' split_by options.")

        # create new document dicts for each text split, add split_id
        i = 0
        content_dicts_list = []
        for txt, tags in zip(text_splits, tag_splits):
            # for i, txt_and_labels in enumerate(zip(text_splits, label_splits)):
            doc = deepcopy(content_dict)
            # txt, labels = txt_and_labels
            doc["content"] = txt
            if "meta" not in doc.keys() or doc["meta"] is None:
                doc["meta"] = {
                    # don't think I can pass a list of strings, so comma separated it is
                    "SPaR_labels": ', '.join(tags)
                }
            else:
                doc["meta"]["SPaR_labels"] = ', '.join(tags)

            doc["meta"]["split_id"] = i
            doc["meta"]["split_size"] = split_length
            content_dicts_list.append(doc)
            i += 1

        return content_dicts_list

    def _ngram(self, seq: str, n: int) -> Generator[str, None, None]:
        """
        Return ngram (of tokens - currently split by whitespace)
        :param seq: str, string from which the ngram shall be created
        :param n: int, n of ngram
        :return: str, ngram as string
        """

        # In order to maintain the original whitespace, but still consider \n and \t for n-gram tokenization,
        # we add a space here and remove it after creation of the ngrams again (see below)
        seq = seq.replace("\n", " \n")
        seq = seq.replace("\t", " \t")

        words = seq.split(" ")
        ngrams = (
            " ".join(words[i: i + n]).replace(" \n", "\n").replace(" \t", "\t") for i in range(0, len(words) - n + 1)
        )

        return ngrams

    def _allngram(self, seq: str, min_ngram: int, max_ngram: int) -> Set[str]:
        lengths = range(min_ngram, max_ngram) if max_ngram else range(min_ngram, len(seq))
        ngrams = map(partial(self._ngram, seq), lengths)
        res = set(chain.from_iterable(ngrams))
        return res

    def _find_longest_common_ngram(
        self, sequences: List[str], max_ngram: int = 30, min_ngram: int = 3
    ) -> Optional[str]:
        """
        Find the longest common ngram across different text sequences (e.g. start of pages).
        Considering all ngrams between the specified range. Helpful for finding footers, headers etc.

        :param sequences: list[str], list of strings that shall be searched for common n_grams
        :param max_ngram: int, maximum length of ngram to consider
        :param min_ngram: minimum length of ngram to consider
        :return: str, common string of all sections
        """
        sequences = [s for s in sequences if s]  # filter empty sequences
        if not sequences:
            return None
        seqs_ngrams = map(partial(self._allngram, min_ngram=min_ngram, max_ngram=max_ngram), sequences)
        intersection = reduce(set.intersection, seqs_ngrams)

        try:
            longest = max(intersection, key=len)
        except ValueError:
            # no common sequence found
            longest = ""
        return longest if longest.strip() else None