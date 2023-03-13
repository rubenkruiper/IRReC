import os.path
import time
from allennlp.predictors.predictor import Predictor as AllenNLPPredictor
from allennlp.models.archival import load_archive
from allennlp.common.util import import_module_and_submodules
import spar_serving_utils as su


import_module_and_submodules("lib")
default_path = "../datavolume/spar_trained_models/debugger_train/model.tar.gz"

if os.path.exists(default_path):
    default_archive = load_archive(default_path)
    predictor = AllenNLPPredictor.from_archive(default_archive)

user_query = ""
print("NOTE:\tTo stop running, simply enter 'quit' as input.\n-------------------------------------------------------")
while user_query != "quit":
    user_query = input("Enter text to be parsed: ")
    start_time = time.time()
    # prepare instance and run model on single instance
    docid = ''                  # ToDo - add doc_id during pre_processing?
    token_list = predictor._dataset_reader.tokenizer.tokenize(user_query)
    instance = predictor._dataset_reader.text_to_instance(docid,
                                                          user_query,
                                                          token_list,
                                                          predictor._dataset_reader._token_indexer)
    res = predictor.predict_instance(instance)
    printable_result = su.parse_spar_output(res, ['obj'])
    print(printable_result)
    print("Parsing took {}".format(time.time() - start_time))