import json, os, random, glob
from collections import Counter
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


random.seed(14)

class ObjectAndContextSelector():
    def __init__(self,
                 all_gold_doc_ids,
                 bert_model_name: str = "bert-base-cased"):
        self.gold_doc_ids = [x.rsplit('/',1)[1] for x in all_gold_doc_ids]

        if bert_model_name is not None:
            self.tokenizer = PretrainedTransformerTokenizer(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name

        self.defined_terms = []

        # { "text for some span" : ["doc_id", "doc_id", ... ], ... }
        self.object_spans = {}
        # { "doc_id" : "sentence"}
        self.sentences = {}

    def get_objects(self, word_list, tag_list, mwe_type='obj'):
        """
        Collects the MWEs found by a trained tagger, handles discontiguous spans.
        """

        mwes = []
        current_mwe = []
        previous_head = []
        just_removed = []

        for idx, t in enumerate(tag_list):
            t_head, t_type = t.split('-')
            if t_type != mwe_type:
                # We're only interested in collecting a specific type of MWE for now, e.g., objects / actions
                if current_mwe != []:
                    # store previously collected mwe parts
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

        return mwes

    def mwe_list_to_string(self, mwe_list):
        text = ''
        for w in mwe_list:
            if w.startswith('##'):
                text += w[2:]
            else:
                text += " " + w

        text = text[1:]
        # if text.startswith('the ') or text.startswith('The '):
        #     text = text[4:]
        # elif text.startswith('a ') or text.startswith('A '):
        #     text = text[2:]
        # elif text.startswith('an ') or text.startswith('An '):
        #     text = text[3:]
        return text

    def grab_definitions(self, file_paths):
        data = {}
        for fp in file_paths:
            f_name = fp.rsplit('/', 1)[1].rsplit('.', 1)[0]
            with open(fp) as f:
                data[f_name] = json.load(f)

        for file_name, sections in data.items():
            for subtitle in sections.keys():
                if subtitle.startswith('Appendix A.'):
                    subsubtitle = [x for x in sections[subtitle].keys() if x != 'url'][0]
                    doc_definitions = [k for k in sections[subtitle][subsubtitle]['definitions'].keys() if k != '']
                    for d in doc_definitions:
                        d = d.lower().strip()

                        if d[0] in ["'", '"'] and d[-1] in ["'", '"']:
                            d = d[1:-1]

                        if '-' in d:
                            # tokenization is different with -
                            d_parts = d.split('-')
                            d = ' - '.join(d_parts)

                        if d not in self.defined_terms:
                            self.defined_terms.append(d)

    def check_if_mwe_is_defined(self, mwe):
        lowercased_mwe = mwe.lower()
        if lowercased_mwe in self.defined_terms:
            # exact matches
            return True
        else:
            # partial matches
            partial_defined_terms = [x for x in self.defined_terms if x in lowercased_mwe]
            if len(partial_defined_terms) > 0:
                return True
            else:
                # no match = good
                return False

    def read_predictions(self, predictions_fp, defined_fps):
        # grab defined terms
        self.grab_definitions(defined_fps)

        # read predictions from output file
        with open(predictions_fp, 'r') as f:
            predictions_list = [json.loads(jsonline) for jsonline in f.readlines()]

        for prediction in predictions_list:
            mask = prediction['mask']
            tag_list = prediction['tags']
            token_list = prediction['words']
            doc_id = prediction['doc_id']

            if doc_id in self.gold_doc_ids:
                # ignore any doc_ids that have been annotated (used in train/dev/test)
                continue

            if prediction["sentence"].startswith("all_figures"):
                # ignore figure references
                continue

            self.sentences[doc_id] = prediction["sentence"]

            # not sure if this is necessary; doesn't seem to be
            predicted_tags = [t for m, t in zip(mask, tag_list) if m]
            mwes = self.get_objects(token_list, predicted_tags, 'obj')

            # Ignore defined terms (may not catch partial
            mwe_strings = [self.mwe_list_to_string(m) for m in mwes]
            non_defined_mwes = [mwe for mwe in mwe_strings if not self.check_if_mwe_is_defined(mwe)]

            for object_span in non_defined_mwes:
                if object_span in self.object_spans.keys():
                    self.object_spans[object_span].append(doc_id)
                else:
                    self.object_spans[object_span] = [doc_id]

    def select_and_write_predictions(self, k=165):
        # Select random objects
        random_objects = random.choices(list(self.object_spans.keys()), k=k)
        # Select random doc_id where this object occurs
        random_doc_ids = []
        for object in random_objects:
             random_doc_ids.append(random.choice(self.object_spans[object]))

        # Grab the sentence, and then write with doc_id and object
        file_idx = 0
        for doc_id, obj in zip(random_doc_ids, random_objects):
            lines_to_write = [
                "Sentence:\n",
                self.sentences[doc_id],
                "\n\nObject identified:\n",
                obj,
                "\n\nDocument id:\n",
                doc_id
            ]
            file_idx += 1

            with open("classifier_data/posthoc_judgement/{}.txt".format(str(file_idx)), 'w') as f:
                f.writelines(lines_to_write)
                f.close()


if __name__ == "__main__":
    # file_paths to grab the definitions
    file_paths = ['../i-ReC/classifier_data/scottish/domestic_standards.json',
                  '../i-ReC/classifier_data/scottish/non-domestic_standards.json']
    predictions_file = 'predictions/all_sentence_predictions.json'
    all_gold_doc_ids = glob.glob('classifier_data/all_annotated/*.txt')
    num_to_select = 165

    object_selector = ObjectAndContextSelector(all_gold_doc_ids)
    object_selector.read_predictions(predictions_file, file_paths)
    object_selector.select_and_write_predictions(num_to_select)
