import json, os, shutil, sys, torch
from argparse import ArgumentParser
from allennlp.commands import main
from spar_lib.evaluation_script import SimpleEvaluator


if __name__ == "__main__":
    argparse = ArgumentParser(description='spar_lib semantic chunking')
    argparse.add_argument('-c', "--config_file", type=str, default="/data/spar_data/experiments/attention_tagger.json")
    argparse.add_argument('-m', "--model_path", default="/data/spar_trained_models/debugger_train")
    argparse.add_argument('-i', "--input_file_path", default="")
    argparse.add_argument('-o', "--output_file_path", default="")
    argparse.add_argument("--batchsize", type=int, default=2)
    argparse.add_argument("--predict", action='store_true')
    argparse.add_argument("--evaluate", action='store_true')

    args = argparse.parse_args()
    config_file = args.config_file
    serialization_dir = args.model_path
    predictions_output = args.output_file_path

    if args.predict:

        sys.argv = [
            "allennlp",
            "predict",
            serialization_dir,
            args.input_file_path,
            "--predictor", "span_tagger",
            "--include-package", "spar_lib",
            "--use-dataset-reader",
            "--batch-size", str(args.batchsize)
        ]

        # Add output-file if specified
        if predictions_output != "":
            sys.argv += ["--output-file", predictions_output]

        main()

    elif args.evaluate:
        # ToDo - Implemented a workaround: 1st) predict 2nd) evaluate predicted output
        #  issue: can't figure out how to use/set the pretrained transformer tokenizer using
        #  AllenNLP, thus running into vocab issues, and have no time rn to sort this out

        if predictions_output == "":
            predictions_output = 'predictions/debug_output.json'

        if os.path.exists(predictions_output):
            # remove the predictions from previous models
            os.remove(predictions_output)

        # First predict ~ default output is location is 'predictions/debug_output.json'
        sys.argv = [
            "allennlp",  # command name, not used by main
            "predict",
            serialization_dir,
            args.input_file_path,
            "--predictor", "span_tagger",
            "--output-file", predictions_output,
            "--include-package", "spar_lib",
            "--use-dataset-reader",
            "--batch-size", str(args.batchsize)
        ]
        main()

        # Then simply run a script to evaluate and print metrics for the output file
        evaluator = SimpleEvaluator(predictions_output, args.input_file_path)
        evaluator.evaluate()


    else:
        # TRAIN
        # Training will fail if the serialization directory already
        # has stuff in it. To re-use the same directory for debugging
        # we clear the contents in advance.
        shutil.rmtree(serialization_dir, ignore_errors=True)

        # Assemble the command into sys.argv
        sys.argv = [
            "allennlp",  # command name, not used by main
            "train",
            config_file,
            "-s", serialization_dir,
            "--include-package", "spar_lib"
        ]

        # Simple overrides to train on CPU if no GPU available, with a possibly smaller batch_size
        if not torch.cuda.is_available():
            overrides = json.dumps({"trainer": {"cuda_device": -1}})    # ,
                                    # "data_loader": {"batch_sampler": {"batch_size": 16}}})
            sys.argv += ["-o", overrides]

        main()
