from typing import List, Union
import os
import imp
import concurrent.futures

from tqdm import tqdm
from pathlib import Path
from textblob import TextBlob
from threading import current_thread

from utils.customdocument import CustomDocument

spartxt_path = Path.cwd().joinpath("SPaR.txt")
with open(spartxt_path.joinpath('spar_predictor.py'), 'rb') as fp:
    spar_predictor = imp.load_module(
        'spar_predictor', fp, 'SPaR.txt.spar_predictor.py',
        ('.py', 'rb', imp.PY_SOURCE)
    )


# These should automatically run on your Nvidia GPU if available
class SparInstance:
    def __init__(self):
        self.sp = spar_predictor.SparPredictor()

    def call(self, input_str: str = ''):
        if input_str:
            # prepare instance and run model on single instance
            docid = input_str  # ToDo - add doc_id during pre_processing?
            token_list = self.sp.predictor._dataset_reader.tokenizer.tokenize(input_str)

            # truncating the input to SPaR.txt to maximum 512 tokens
            token_length = len(token_list)
            if token_length > 512:
                token_list = token_list[:511] + [token_list[-1]]
                token_length = 512

            instance = self.sp.predictor._dataset_reader.text_to_instance(docid, input_str, token_list,
                                                                          self.sp.predictor._dataset_reader._token_indexer)
            result = self.sp.predictor.predict_instance(instance)
            printable_result = self.sp.parse_output(result, ['obj'])
            return {
                "prediction": printable_result,
                "num_input_tokens": token_length,
            }

        # If the input is None, or too long, return an empty list of objects
        return {
            "prediction": {'obj': []},
            "num_input_tokens": 0
        }


class TermExtractor:
    def __init__(self, split_length=300, max_num_cpu_threads=4):
        """
        Initialise `max_num_cpu_threads` separate SPaR.txt predictors
        """
        self.split_length = split_length  # in number of tokens
        self.max_num_cpu_threads = max_num_cpu_threads
        self.PREDICTORS = []
        for i in range(max_num_cpu_threads + 1):
            self.PREDICTORS.append(SparInstance())

    def process_sentence(self, sentence: str = ''):
        """
        """
        predictor_to_use = int(current_thread().name.rsplit('_', 1)[1])
        spartxt = self.PREDICTORS[predictor_to_use]

        # SPaR doesn't handle ALL uppercase sentences well, which the OCR system sometimes outputs
        sentence = sentence.lower() if sentence.isupper() else sentence
        prediction_dict = spartxt.call(sentence)
        if not prediction_dict:
            return []

        pred_labels = prediction_dict["prediction"]
        return pred_labels['obj']

    def split_into_sentences(self, to_be_split: Union[str, List[str]]) -> List[str]:
        """
        """
        if type(to_be_split) == str:
            if ';' in to_be_split:
                # some of the WikiData definitions contain multiple definitions separated by ';'
                to_be_split = to_be_split.split(';')
            else:
                to_be_split = [to_be_split]

        sentences = []
        for text in to_be_split:
            for part in text.split('\n'):
                # split into sentences using PunktSentTokenizer (TextBlob implements NLTK's version under the hood)
                sentences += [str(s) for s in TextBlob(part.strip()).sentences if len(str(s)) > 10]
        return sentences

    def process_sentences(self, sentences: List[str]):
        """
        """
        spar_objects = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_num_cpu_threads) as executor:
            futures = [executor.submit(self.process_sentence, sentences[idx]) for idx in range(len(sentences))]
            spar_objects += [tag for f in futures for tag in f.result()]
        return spar_objects

    def process_custom_document(self, input_document: CustomDocument):
        """
        """
        print(f"Working on: {input_document.source_fp}")
        content_as_list_of_dicts = input_document.to_list_of_dicts()
        total_number_of_sentences_found = 0
        content_idx = 0
        for content_dict in tqdm(content_as_list_of_dicts):

            text = ' '.join([x for x in content_dict["content"].split(' ') if x != ''])
            # some really long paragraphs in the EU regulations are summations that should be split at ';'
            if len(text) > 3000:
                text = text.replace(";", ".\n")

            # We'll split into sentences even if this has been done before, it doesn't take long
            sentences = self.split_into_sentences(text)
            content_dict["meta"]["sentences"] = '###'.join(sentences)
            total_number_of_sentences_found += len(sentences)

            # process sentences in the content and add SPaR.txt object tags to the content dict.
            if not content_dict["meta"]["SPaR_labels"]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_num_cpu_threads) as executor:
                    futures = [executor.submit(self.process_sentence, sentences[idx]) for idx in range(len(sentences))]

                content_spar_objects = [f.result() for f in futures]
                content_dict["meta"]["SPaR_labels"] = ', '.join([tag for tags in content_spar_objects for tag in tags])

            # immediately update the list of content_dicts and every X iterations we save the file
            content_as_list_of_dicts[content_idx] = content_dict
            if content_idx // 5 == 0:
                input_document.replace_contents(content_as_list_of_dicts)
                input_document.write_document()

            content_idx += 1

        print(f"Number of sentences found: {total_number_of_sentences_found}")
        input_document.replace_contents(content_as_list_of_dicts)
        input_document.write_document()
