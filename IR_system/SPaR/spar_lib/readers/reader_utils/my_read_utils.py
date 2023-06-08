import copy
from typing import List, Tuple, Set


def annotations_as_dict(ann_file, original_sentence):
    """
    Reads the annotations from an ann_file and stores them in span_buffer
    :param      ann_file:       BRAT annotations
    :return:    span_buffer:    dict holding the annotation indices (possibly discontiguous) and original text
    """
    span_idx = 0
    span_buffer = {}

    with open(ann_file) as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if line.startswith('E'):
            pass  # For the span-pair classification we don't handle Events
        elif line.startswith('A'):
            pass  # Currently we don't care about confidence
        elif line.startswith('R'):
            pass  # For the span-pair classification we don't handle relations
        elif line.startswith('T'):
            # Spans (T) are annotated chunks of text, possibly discontiguous
            span_id, span_and_type, span_text = line.rstrip().split('\t')

            if ';' in span_and_type:
                # Discontiguous spans
                span_type, span_start, gap, span_end = span_and_type.split(' ')
                gap_start, gap_end = gap.split(';')

                ss, se, es, ee = int(span_start), int(gap_start), int(gap_end), int(span_end)

                # re-order of spans if annotation order is reversed
                if ss > ee:
                    ss_, se_ = es, ee
                    es, ee = ss, se
                    ss, se = ss_, se_

                if ss > ee or ss > se or ss > es or se > es or se > ee or es > ee:
                    raise ValueError(f"Issue with the order of span indices; ({ss}, {se}, {es}, {ee}).")

                span_buffer[span_idx] = {'span_id': span_id,
                                         'span_type': span_type,
                                         'span_start': ss,
                                         'gap_start': se,
                                         'gap_end': es,
                                         'span_end': ee,
                                         'span_text': span_text,
                                         'context': original_sentence}
                span_idx += 1
            else:
                # Contiguous spans
                span_type, ss, se = span_and_type.split(' ')
                span_buffer[span_idx] = {'span_id': span_id,
                                         'span_type': span_type,
                                         'span_start': int(ss),
                                         'gap_start': int(se),
                                         'gap_end': int(se),
                                         'span_end': int(se),
                                         'span_text': span_text,
                                         'context': original_sentence}
                span_idx += 1

    return span_buffer


def brat_indices_to_token_indices(brat_start, brat_end, token_list):
    """ Simple conversion per brat idx to token idx """
    bert_ss = -1
    bert_ee = -1
    for list_idx, t in enumerate(token_list):
        if t.idx == brat_start:
            bert_ss = list_idx
        if t.idx_end == brat_end:
            bert_ee = list_idx
    if bert_ss == -1 or bert_ee == -1:
        print("brat_idx and token indices mismatch? Something wrong!")
    return bert_ss, bert_ee


def brat_to_PretainedTransformerTokenizer(token_list, span_buffer):
    """ Convert brat indices to BERT indices """
    # sometimes you don't want to update the same dict with new indices
    bert_indexed_spans = copy.deepcopy(span_buffer)

    for span_idx, span in span_buffer.items():
        span_start_idx = span['span_start']
        gap_start_idx = span['gap_start']
        gap_end_idx = span['gap_end']
        span_end_idx = span['span_end']

        if gap_start_idx == span_end_idx:
            # contiguous span
            ss, ee = brat_indices_to_token_indices(span_start_idx, span_end_idx, token_list)
            bert_indexed_spans[span_idx]['span_start'] = ss
            bert_indexed_spans[span_idx]['gap_start'] = ee
            bert_indexed_spans[span_idx]['gap_end'] = ee
            bert_indexed_spans[span_idx]['span_end'] = ee
        else:
            # discontiguous spans
            one_start, one_end = brat_indices_to_token_indices(span_start_idx, gap_start_idx, token_list)
            two_start, two_end = brat_indices_to_token_indices(gap_end_idx, span_end_idx, token_list)
            bert_indexed_spans[span_idx]['span_start'] = one_start
            bert_indexed_spans[span_idx]['gap_start'] = one_end
            bert_indexed_spans[span_idx]['gap_end'] = two_start
            bert_indexed_spans[span_idx]['span_end'] = two_end

    return bert_indexed_spans


def compute_tags_for_spans(span_buffer, token_list):
    """
    Objects (BH-obj, IH-obj, BD-obj, ID-obj)
    Actions (BH-act, IH-act, BD-act, ID-act)
    Functional and Discourse (BH-func, IH-func, BH-dis, IH-dis)
    """
    none_token = 'PD-pad'
    begin_head = 'BH'
    inside_head = 'IH'
    begin_discontiguous = 'BD'
    inside_discontiguous = 'ID'
    object_type = 'obj'
    action_type = 'act'
    discourse_type = 'dis'
    function_type = 'func'

    tag_list = [none_token] * len(token_list)

    # loop over spans and assign labels to positions - currently not used span_id
    for _, span in span_buffer.items():
        span_start_idx = span['span_start']
        gap_start_idx = span['gap_start']
        gap_end_idx = span['gap_end']
        span_end_idx = span['span_end']

        # determine tags for the span_tokens
        span_type = span['span_type']
        if span_type == 'Functional_span':
            first_token_tag = begin_head + '-' + function_type
            further_tags = inside_head + '-' + function_type
        elif span_type == 'Discourse_span':
            first_token_tag = begin_head + '-' + discourse_type
            further_tags = inside_head + '-' + discourse_type
        elif span_type == 'Object_span':
            first_token_tag = begin_head + '-' + object_type
            further_tags = inside_head + '-' + object_type
        elif span_type == 'Action_span':
            first_token_tag = begin_head + '-' + action_type
            further_tags = inside_head + '-' + action_type

        if tag_list[span_start_idx] != none_token and tag_list[span_start_idx] != first_token_tag:
            # Figure out why a token receives two different tags, this shouldn't happen
            print('Annotation issue in:\n{}'.format(span))

        tag_list[span_start_idx] = first_token_tag
        if span_start_idx != gap_start_idx:
            tag_list[span_start_idx + 1:gap_start_idx + 1] = [further_tags] * (gap_start_idx - span_start_idx)

        if gap_start_idx != span_end_idx:
            # discontiguous span
            if span_type == 'Object_span':
                first_token_tag = begin_discontiguous + '-' + object_type
                further_tags = inside_discontiguous + '-' + object_type
            elif span_type == 'Action_span':
                first_token_tag = begin_discontiguous + '-' + action_type
                further_tags = inside_discontiguous + '-' + action_type

            # if tag_list[span_start_idx] != none_token:
            #     # Figure out why a token receives two different tags, this shouldn't happen
            #     print('Annotation issue in:\n{}'.format(span))
            tag_list[gap_end_idx] = first_token_tag
            if gap_end_idx != span_end_idx:
                tag_list[gap_end_idx + 1:span_end_idx + 1] = [further_tags] * (span_end_idx - gap_end_idx)

    return tag_list


def order_annotations_for_file(span_buffer):
    # ToDo - is the ordering necessary? Probably have to re-write
    # order the spans by occurrence in the sentence
    ordered_spans = sorted(span_buffer.values(), key=lambda item: item['span_end'])
    ordered_span_buffer = {}
    for idx, v in enumerate(ordered_spans):
        ordered_span_buffer[idx] = v
    return ordered_span_buffer


def get_annotations_from_ann_file(ann_file, original_text, token_list):
    """
    Stores the annotations from an .ann file into the following buffers, then stores
    them in our json format.
    """
    span_buffer = annotations_as_dict(ann_file, original_text)
    ordered_span_buffer = order_annotations_for_file(span_buffer)

    # convert brat indices to BERT indices
    ordered_span_buffer = brat_to_PretainedTransformerTokenizer(token_list, ordered_span_buffer)

    # compute and return the tags
    return compute_tags_for_spans(ordered_span_buffer, token_list)


TypedStringSpan = Tuple[str, Tuple[int, int]]


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return " ".join(self.tag_sequence)

def discontiguous_tags_to_spans(tag_sequence: List[str], types_to_ignore: List[str] = None) -> List[TypedStringSpan]:
    """
    Adapted from allennlp.classifier_data.dataset_readers.dataset_utils.bio_tags_to_spans
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans.

    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.

    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    types_to_ignore = types_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_type = None
    for index, string_tag in enumerate(tag_sequence):
        current_tag = string_tag[:2]
        current_type = string_tag[3:]

        if current_tag not in ["PD", "BH", "IH", "BD", "ID"]:
            raise InvalidTagSequence(tag_sequence)

        if current_tag == "PD" or current_type in types_to_ignore:
            # The span has ended.
            if active_type is not None:
                spans.add((active_type, (span_start, span_end)))
            active_type = None
            # We don't care about types we are
            # told to ignore, so we do nothing.
            continue
        elif current_tag in ("BH", "BD"):
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_type is not None:
                spans.add((active_type, (span_start, span_end)))
            active_type = current_type
            span_start = index
            span_end = index
        elif current_tag in ("IH", "ID") and current_type == active_type:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case when a label is an "IH" or "ID, but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different annotation type
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_type is not None:
                spans.add((active_type, (span_start, span_end)))
            active_type = current_type
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_type is not None:
        spans.add((active_type, (span_start, span_end)))
    return list(spans)



# def no_gaps_in_annotations_check(span_buffer, doc_name):
#     """
#     Check that every word or punctuaction has been annotated. This can be hard to see in BRAT sometimes.
#     The order of annotations doesn't always match the order of the actual text spans / tokens.
#     """
#     all_spans = []
#     for span in span_buffer:
#         if span_buffer[span]['gap_start'] != '':
#             s = int(span_buffer[span]['span_start'])
#             gs = int(span_buffer[span]['gap_start'])
#             ge = int(span_buffer[span]['gap_end'])
#             e = int(span_buffer[span]['span_end'])
#             text = span_buffer[span]['text']
#
#             part_one = (s, gs, text[s:gs])
#             part_two = (ge, e, text[gs+1:])
#
#             all_spans.append(part_one)
#             all_spans.append(part_two)
#
#         else:
#             all_spans.append((int(span_buffer[span]['span_start']),
#                               int(span_buffer[span]['span_end']),
#                               span_buffer[span]['text']))
#     issues = False
#
#     if all_spans == []:
#         print("File has no annotations: {}".format(doc_name))
#         return True
#
#     # discontiguous spans may share a start_idx, but not the end_idx with other spans
#     # e.g. [standards] 2.4 and 2.5 --> [standards 2.4] and [standards (gap) 2.5]
#     unique_starts = []
#     ends_for_s = {}
#     for s, e, _ in all_spans:
#         if s not in ends_for_s:
#             ends_for_s[s] = e
#         else:
#             ends_for_s[s] = max(e, ends_for_s[s])
#         if s not in unique_starts:
#             unique_starts.append(s)
#
#     max_end = max(ends_for_s.values())
#     for e in ends_for_s.values():
#         # doesn't check the very last span
#         if e != max_end:
#             # +1 if there's a space between words, otherwise +0
#             if (e + 1 not in unique_starts) and (e not in unique_starts):
#                 issues = True
#                 # note that if the .txt file contains multiple spaces, the brat indices are spaced out further
#                 print('{} Annotation may have gaps at span ending with idx:\t{}'.format(doc_name, e))
#
#     return issues


# def brat_to_spacy_tokens(doc, span_buffer):
#     # CONVERT SPAN INDICES TO SPACY TOKEN INDICES - not using spacy right anymore
#     for token in doc:
#         for span in span_buffer:
#             span_start_idx = int(span_buffer[span]['span_start'])
#             spand_end_idx = int(span_buffer[span]['span_end'])
#             if token.idx == span_start_idx:
#                 span_buffer[span]['span_start'] = str(token.i)
#             if token.idx + len(token.text) == spand_end_idx:
#                 span_buffer[span]['span_end'] = str(token.i)
#     return span_buffer
