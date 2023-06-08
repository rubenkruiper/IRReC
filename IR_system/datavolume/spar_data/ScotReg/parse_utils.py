import spacy
# import benepar

def setup_constituency_parser(const_parse=False):
    # set up spacy and benepar if
    if const_parse:
        nlp = spacy.load('en_core_web_md', disable=['parser'])
        # avoid sentence-splitting based on dependency parser
        nlp.add_pipe(nlp.create_pipe("sentencizer"))
        if spacy.__version__.startswith('2'):
            nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            nlp.add_pipe("benepar", config={"model": "benepar_en3"})
        return nlp
    else:
        return spacy.load('en_core_web_md')


def get_tokens(doc):
    return [t for t in doc]


def get_sentences(doc):
    return list(doc.sents)


def get_spacy_noun_chunks(doc):
    noun_chunks = []
    for noun_chunk in doc.noun_chunks:
        noun_chunks.append(noun_chunk.text)
    return noun_chunks


def get_constituency_labels(sent):
    constituency_tree = sent._.parse_string
    return constituency_tree
