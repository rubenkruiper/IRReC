import json, glob
from typing import List, Dict, Any
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from lib.readers.reader_utils.my_read_utils import *
from sklearn import metrics


class SimpleEvaluator():
    """
    Ugly temporary solution to evaluate a model; compares the predicted output against the gold input
    Reason was that evaluation of the tagger is not very flexible/adaptable in AllenNLP (if I remember correctly).
    """

    def __init__(self,
                 predictions_fp,
                 gold_fp,
                 bert_model_name: str = "bert-base-cased"):
        self.predictions_input = predictions_fp
        self.gold_input = gold_fp

        if bert_model_name is not None:
            self.tokenizer = PretrainedTransformerTokenizer(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name

    def read_gold(self):
        text_files = sorted(glob.glob(self.gold_input + "*.txt"))
        ann_files = sorted(glob.glob(self.gold_input + "*.ann"))

        gold_annotations = []
        for text_file, ann_file in zip(text_files, ann_files):
            doc_name = text_file.rsplit('/', 1)[1].rsplit('.', 1)[0]

            with open(text_file, "r") as tf:
                original_sentence = tf.read()

            token_list = self.tokenizer.tokenize(original_sentence)
            tag_list = get_annotations_from_ann_file(ann_file, original_sentence, token_list)
            gold_annotations.append({"sent_id": doc_name, "sentence": original_sentence,
                                     "token_list": [t.text for t in token_list], "tag_list": tag_list})
        return gold_annotations


    def evaluate(self):

        # read gold classifier_data from 'predictions_input'
        gold_instances = self.read_gold()

        # read predictions from output file
        with open(self.predictions_input, 'r') as f:
            predictions_list = [json.loads(jsonline) for jsonline in f.readlines()]

        # sklearn test
        y_true = []
        y_pred = []

        for prediction in predictions_list:
            mask = prediction['mask']
            tag_list = prediction['tags']
            token_list = prediction['words']

            for gold in gold_instances:
                # should be same order for predict, at least only 1 match
                if gold['token_list'] == token_list:
                    # compare tag_lists
                    predicted_tags = [t for m, t in zip(mask, tag_list) if m]
                    # measure(predicted_tags, gold["tag_list"])
                    y_true += gold["tag_list"][1:-1]
                    y_pred += predicted_tags[1:-1]

        # p, r, f = measure.get_metric()
        # print("Overall precision: {:.4f}, recall: {:.4f}, F1: {:.4f}".format(p, r, f))

        # check using sklearn
        print(metrics.classification_report(y_true, y_pred, digits=4))
        # print("sklearn P: {}".format(metrics.precision_score(y_true, y_pred, average='macro')))
        # print("sklearn R: {}".format(metrics.recall_score(y_true, y_pred, average='macro')))
        # print("sklearn F1: {}".format(metrics.f1_score(y_true, y_pred, average='macro')))

if __name__ == "__main__":
    evaluator = SimpleEvaluator("../predictions/test_predictions.json",
                                "../classifier_data/test/",
                                "bert-base-cased")
    evaluator.evaluate()

