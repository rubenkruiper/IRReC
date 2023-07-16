from typing import List, Dict

from overrides import overrides
import numpy
import pandas as pd

from allennlp.common.util import JsonDict
from allennlp.data import Instance, DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import FlagField, TextField, SequenceLabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


@Predictor.register("span_tagger")
class SpanTaggerPredictor(Predictor):
    """
    Adapted from SentenceTaggerPredictor (allennlp.predictors.sentence_tagger)
    --> Only changed the tokenizer in __init__
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader,
                 bert_model_name: str ="bert-base-cased"):   # todo  "whaleloops/phrase-bert"
        super().__init__(model, dataset_reader)
        self._tokenizer = PretrainedTransformerTokenizer(bert_model_name)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"words"` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)
