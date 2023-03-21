from fastapi import FastAPI, Query
from pydantic import Required

## set up SPaR predictor
from typing import Optional
from spar_predictor import SparPredictor
from allennlp.common.util import import_module_and_submodules
import spar_serving_utils as su

# Set up a SPaR.txt predictor
import_module_and_submodules("spar_lib")
default_path = "/data/spar_trained_models/debugger_train/model.tar.gz"

spar_predictor = SparPredictor(default_path)
predictor = spar_predictor.predictor
## Set up the API
SPaR_api = FastAPI()


# @SPaR_api.get("/tokenize/{input_str}")
# def tokenize(input_str: Optional[str] = Query(default=Required, max_length=1000)):
#     return predictor._dataset_reader.tokenizer.tokenize(input_str)
#


@SPaR_api.get("/predict_objects/{input_str}")
def predict_objects(input_str: Optional[str] = Query(default=Required, max_length=1000)):
    """
    Predict the object in a given string (expecting a single sentence usually).
    """
    if input_str:
        # prepare instance and run model on single instance
        docid = ''  # ToDo - add doc_id during pre_processing?
        token_list = spar_predictor.predictor._dataset_reader.tokenizer.tokenize(input_str)

        # truncating the input to SPaR.txt to maximum 512 tokens
        token_length = len(token_list)
        if token_length > 512:
            token_list = token_list[:511] + [token_list[-1]]
            token_length = 512

        instance = spar_predictor.predictor._dataset_reader.text_to_instance(
            docid, input_str, token_list, spar_predictor.predictor._dataset_reader._token_indexer
        )
        res = predictor.predict_instance(instance)
        printable_result, spans_token_length = su.parse_spar_output(res, ['obj'])
        return {
            "prediction": printable_result,
            "num_input_tokens": token_length,
            "num_output_tokens": spans_token_length
        }
    # If the input is None, or too long, return an empty list of objects
    return {
            "prediction": {'obj': []},
            "num_input_tokens": 0,
            "num_output_tokens": 0
        }

