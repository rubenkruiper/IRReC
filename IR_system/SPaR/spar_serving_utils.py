from typing import List

"""
Several classes and methods to keep track of SPaRtxt outputs for further processing.
"""


# ----------------------- Classes  -----------------------#
class Indices(object):
    """
    A container for the indices of a span
    """
    def __init__(self,
                 contiguous_start: int = -1,
                 contiguous_end: int = -1,
                 discontiguous_start: int = -1,
                 discontiguous_end: int = -1):
        self.ss = contiguous_start
        self.se = contiguous_end
        self.es = discontiguous_start
        self.ee = discontiguous_end

    def __str__(self):
        return str([self.ss, self.se, self.es, self.ee])

    def to_string(self):
        return str([self.ss, self.se, self.es, self.ee])

    def set_ss(self, idx: int):
        self.ss = idx

    def set_se(self, idx: int):
        self.se = idx

    def set_es(self, idx: int):
        self.es = idx

    def set_ee(self, idx: int):
        self.ee = idx

    def is_discontiguous(self):
        if self.se != self.ee and self.ee > 0:
            return True
        return False


class SingleSpan(object):
    """
    Immutable span object
    """
    def __init__(self, span: str, span_type: str):
        # Text covered by span and type
        self.span = span
        self.span_type = span_type

    def __str__(self):
        if self:
            return self.span
        return ''

    def to_list(self):
        return [self.span, self.span_type]


class Sentence(object):
    """
    Immutable sentence object to hold a single sentence
    """
    def __init__(self, sentence: str):
        self.sentence = sentence
        self.spans = {}

    def __str__(self):
        if self:
            return self.sentence
        return ''

    def to_dict(self):
        """ ToDO; might want to change how this object is formatted as a dict """
        return {self.sentence: self.spans}

    def add_span(self, span: str, indices: int):
        self.spans[span] = indices

    def get_indices(self, span: str):
        if span not in self.spans:
            raise ValueError(f"'{span}' not in this sentence")
        return self.spans[span]


# ----------------------- Functions  -----------------------#
def mwe_list_to_string(mwe_list: List[str]):
    """
    Converts a list of tokens to a single word.

    :param mwe_list:    List of tokens.
    :return text:   String representation reconstructed from the list of tokens.
    :return mwe_token_length:   Token length of the mwe_list.
    """
    text = ''
    mwe_token_length = 0
    for w in mwe_list:
        mwe_token_length += 1
        if w.startswith('##'):
            text += w[2:]
        else:
            text += " " + w

    if text:
        text = text[1:]
        return text, mwe_token_length


def update_indices(indices_obj: Indices):
    if indices_obj.se == -1:
        indices_obj.se = indices_obj.ss
    if indices_obj.es == -1:
        indices_obj.es = indices_obj.se
    if indices_obj.ee == -1:
        indices_obj.ee = indices_obj.es


def get_spans(token_list: List[str], tag_list: List[str], mwe_type: str) -> (List[str], List[Indices], List[int]):
    """
    Helper function to collect the MWE spans found by a trained SPaR.txt tagger, handles discontiguous spans.

    :param token_list:   List of tokens for some input sentence.
    :param tag_list:   List of predicted tags for the given token_list.
    :param mwe_type:    Types of spans that we're isolating from the text.
    :return mwe_strings:    List of spans as strings, reconstructed from the tokens.
    :return indices:    List of character-level indices for each span (potentially discontiguous), indicating where
                        in the source text the span was found.
    :return mwe_lengths:    List of numbers of token for each span.
    """

    mwes = []
    indices = []

    current_mwe = []
    current_indices = Indices()

    previous_head = []
    previous_indices = Indices()
    just_removed = []

    for idx, t in enumerate(tag_list):
        # idx = t_id - 1
        t_head, current_type = t.split('-')

        if current_type != mwe_type:
            # We're collecting a specific type of MWE each time, e.g., objects / actions, so we can disentangle them
            if current_mwe:
                # store mwe and corresponding indices
                mwes.append(current_mwe)
                indices.append(current_indices)
                # In discontiguous cases, we want to store current_mwe into previous head
                if not just_removed:
                    # if we just encountered a BD-mwe_type, then previous_head should not change
                    previous_head = current_mwe
                    previous_indices = current_indices
            current_mwe = []
            current_indices = Indices()
            continue

        if t_head == 'BH':
            # We expect a completely new span of type `mwe_type'
            just_removed = []

            if not current_mwe:
                # Start collecting a new mwe
                current_mwe = [token_list[idx]]
                current_indices = Indices(idx)
            else:
                # Store old mwe and start collecting a new mwe
                mwes.append(current_mwe)
                indices.append(current_indices)
                previous_head = current_mwe
                previous_indices = current_indices
                current_mwe = [token_list[idx]]
                current_indices = Indices(idx)
        elif t_head == 'IH':
            # Continue collecting the same object, each time updating the head's end index
            # Due to tagging inaccuracy, this may continue from a BH or a BD
            current_mwe.append(token_list[idx])
            if current_indices.es == -1:
                current_indices.set_se(idx)
            else:
                current_indices.set_ee(idx)
        elif t_head == 'BD':
            # Case: BD-x with a preceding BH-x (note x is a given type)
            if just_removed != previous_head and previous_head in mwes:
                # Remove BH-based previous_head from mwes,
                mwes.reverse()        # ToDo; The last mwe may not be the mwe we want to remove ? otherwise simply pop()
                mwes.remove(previous_head)
                mwes.reverse()
                indices.reverse()
                indices.remove(previous_indices)
                indices.reverse()
                just_removed = previous_head
                #  and prepend it to the current_mwe, also setting the current indices to previous with new .es
                current_mwe = just_removed + [token_list[idx]]
                current_indices = previous_indices
                current_indices.set_es(idx)
                # reset previous head, considering that it has been used definitely now.
                previous_head = []

            else:   # Case: BD-x without a preceding BH-x (note x is a given type)
                # This CAN occur based on allowed transitions in the CRF, but shouldn't happen often
                if current_mwe:
                    # Store current mwe
                    mwes.append(current_mwe)
                    indices.append(current_indices)
                    previous_head = current_mwe
                    previous_indices = current_indices
                # start new mwe
                current_mwe = [token_list[idx]]
                current_indices = Indices(idx)
        elif t_head == 'ID':
            # Continue collecting discontiguous, each time updating the end's end index
            current_mwe.append(token_list[idx])
            current_indices.set_ee(idx)

    if mwes:
        mwe_strings, mwe_lengths = zip(*[mwe_list_to_string(mwe) for mwe in mwes if mwe])
        [update_indices(i) for i in indices]
    else:
        mwe_strings, mwe_lengths = [''], [0]

    return mwe_strings, indices, mwe_lengths


def parse_spar_output(prediction: dict, span_types: List[str] = ['obj', 'act', 'func', 'dis']) -> (dict, int):
    """
    SPaR.txt outputs are formatted following the default AllenNLP json structure. That is, a list of tokens and
    a list of the corresponding SPaR.txt tags that were predicted. This function reconstructs the text that belongs
    to each of the predicted spans.

    :param prediction:  Dictionary loaded from SPaR.txt output.
    :params span_types: The types of spans that we'd like to keep. SPaR.txt predicts objects `'obj'`, actions`'act'`,
                        functions `'func'`, and discourse `'dis'` spans.
    :return NER:    Dictionary holding lists of spans per type `{ 'obj': [], 'dis': [], 'func': [], 'act': [], }`.
    :return total_number_of_tokens_in_selected_spans:   A count of the token-lengths of all retained spans.
    """
    # read predictions from file
    sentence = prediction["sentence"]
    sent_id = prediction["doc_id"]

    current_sent_obj = Sentence(sentence)

    mask = prediction['mask']
    tag_list = prediction['tags']
    token_list = prediction['words']

    total_number_of_tokens_in_selected_spans = 0
    predicted_tags = [t for m, t in zip(mask, tag_list) if m]
    for mwe_type in span_types:
        list_of_spans, list_of_indices_lists, list_of_token_lengths = get_spans(token_list, predicted_tags, mwe_type)
        total_number_of_tokens_in_selected_spans += sum(list_of_token_lengths)
        for span, indices in zip(list_of_spans, list_of_indices_lists):
            current_span = SingleSpan(span, mwe_type)
            # store by indices, because spans may contain the same text
            current_sent_obj.spans[indices.to_string()] = current_span.to_list()

    # todo: clean these stitched together scripts at some point in my life :)
    sent_and_spans = current_sent_obj.to_dict()
    sentence = list(sent_and_spans.keys())[0]
    spans_dict = sent_and_spans[sentence]

    NER = {}
    for key in span_types:
        NER[key] = []

    for indices_str, span_tuple in spans_dict.items():
        # Indices, here, refers to the (possibly discontiguous) indices of the span in the sentence:
        # [start_start, start_end, end_start, end_end]
        ### todo; could use indices to count number of discontiguous spans
        # indices = [int(i) for i in indices_str[1:-1].split(',')]
        # assert len(indices) == 4
        span, span_type = span_tuple
        NER[span_type].append(span)

    return NER, total_number_of_tokens_in_selected_spans
