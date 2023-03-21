import itertools
import json
import os
import collections
from pathlib import Path
from typing import List, Union


class Content(object):
    def __init__(self, output_filepath: Path, source_filepath: Path,
                 doc_name: str, text: str, page_nr: int, split_id: int, split_size: int = -1):
        """
        A content object keeps track of the fields that we want to index.

        :param output_filepath:    Path where the converted document is stored.
        :param source_filepath:    Path to the input PDF document.
        :param :    Filename of the document.
        :param text:    Part of the document's text to be indexed and retrieved, and in which relevant labels are found.
        :param page_nr: Page number within the source document, where the text can be found.
        :param split_id:    Indicates n-th split on a page within the source document, where the text can be found. If
                            the number is negative, then the document has only been converted -- not preprocessed and
                            split yet.
        :param split_size:  The size of splits is stored in each `Content`, so the `Document` can access it later.
        """
        self.output_filepath = output_filepath
        self.source_filepath = source_filepath
        self.text = text
        self.id = doc_name + "##" + str(page_nr)

        self.split_id = split_id
        if split_id != -1:
            self.id += "##" + str(split_id)

        self.split_size = split_size

        self.doc_title = ''
        self.sentences = []
        self.NER_labels = []
        self.filtered_NER_labels = []
        self.filtered_NER_labels_domains = []
        self.neighbours = []

    @classmethod
    def from_dict(cls, haystack_dict):
        """
        A content object keeps track of the fields that we want to index. This constructor initializes
        the object from a dict.
        """
        content = cls(Path(haystack_dict['meta']['output_filepath']),  # dict contains str, convert to Path
                      Path(haystack_dict['meta']['source_filepath']),
                      haystack_dict['id'].split("##")[0],  # doc_name is unchanged
                      haystack_dict['content'],  # text
                      int(haystack_dict['id'].split("##")[1]),  # page_nr is unchanged
                      int(haystack_dict['meta']['split_id']),  # split_id
                      int(haystack_dict['meta']['split_size'])  # split_size
                      )
        content.doc_title = haystack_dict['meta']['doc_title']
        content.sentences = haystack_dict['meta']['sentences'].split("###")
        content.NER_labels = haystack_dict['meta']['NER_labels'].split(", ")
        content.filtered_NER_labels = haystack_dict['meta']['filtered_NER_labels'].split(", ")
        content.filtered_NER_labels_domains = haystack_dict['meta']['filtered_NER_labels_domains'].split(", ")
        content.neighbours = haystack_dict['meta']['neighbours'].split(", ")
        return content

    def to_dict(self):
        if any(type(a) == list for a in self.filtered_NER_labels_domains):
            self.filtered_NER_labels_domains = list(itertools.chain(*self.filtered_NER_labels_domains))
        if any(type(a) == list for a in self.neighbours):
            self.neighbours = list(itertools.chain(*self.neighbours))

        haystack_dict = {
            "content": self.text,
            "content_type": "text",
            "id": self.id,
            "meta": {
                "text": self.text,  # keep track of text when indexing other fields
                "doc_title": self.doc_title,
                "split_size": self.split_size,
                "split_id": self.split_id,
                "source_filepath": str(self.source_filepath),  # convert Path obj to str
                "output_filepath": str(self.output_filepath),
                "sentences": '###'.join(self.sentences),
                "NER_labels": ', '.join(self.NER_labels),
                "filtered_NER_labels": ', '.join(self.filtered_NER_labels),
                "filtered_NER_labels_domains": ', '.join(self.filtered_NER_labels_domains),
                "neighbours": ', '.join(self.neighbours),
            }
        }
        return haystack_dict

    def set_doc_title(self, doc_title):
        self.doc_title = doc_title

    def set_id(self, new_id):
        self.id = new_id

    def set_ner_labels(self, labels: List = [str]):
        self.NER_labels = labels

    def set_filtered_ner_labels(self, filtered_labels: List = [str]):
        self.filtered_NER_labels = filtered_labels

    def set_filtered_ner_label_domains(self, NER_label_domains: List = [str]):
        self.filtered_NER_labels_domains = NER_label_domains

    def set_neighbours(self, neighbours: List = [str]):
        self.neighbours = neighbours


class CustomDocument(object):
    def __init__(self,
                 output_filepath: Path,
                 source_filepath: Path = None,
                 split_size: int = 100):
        """
        NOTE: Haystack has its own `Document` class -- hence named CustomDocument from now on...

        Documents are split up into parts of `self.split_size` (measured in number of words). This class keeps track of
        an entire document's text-parts in the form of `Content` objects. We're mostly interested in keeping track
        of specific fields inside the Document's contents ~ labels that we've identified during preprocessing etc. The
        class also provides some functionality like storing to file and loading a Document from file.

        :param output_filepath:    Path (`str`) where the Document will be stored as a list of content dicts.
        :param source_filepath:    Path (`str`) to the original source document.
        :param split_size:  Size of content that will be written to the `DocumentStore`.
        """
        self.output_fp = output_filepath
        self.source_fp = source_filepath
        if source_filepath:
            self.doc_name = source_filepath.stem

        self.split_size = split_size
        self.all_contents = []  # todo would a set be faster? And do I need to keep order?

    @classmethod
    def load_document(cls, output_fp):
        with open(output_fp, 'r') as f:
            all_contents = [Content.from_dict(json.loads(l)) for l in f.readlines()]
        try:
            # index to first nonempty dict
            i = next(idx for idx, d in enumerate(all_contents) if d)
            doc = cls(output_fp, all_contents[i].source_filepath, all_contents[i].split_size)
            doc.add_contents(all_contents)
            return doc
        except StopIteration:
            # Todo - no contents found in entire output_fp, maybe remove output_fp?
            print("[SKIPPING] Probably an empty file: {}".format(output_fp))
            os.remove(output_fp)
            print("[SKIPPING] Successfully removed as input: {}".format(output_fp))
            return None

    def write_document(self):
        with open(self.output_fp, 'w') as f:
            f.writelines([json.dumps(c.to_dict()) + "\n" for c in self.all_contents])

    def add_content(self, text: str, page_nr: int, doc_title: str, split_id: int = -1):
        content = Content(self.output_fp, self.source_fp, self.doc_name, text, page_nr, split_id, self.split_size)
        content.doc_title = doc_title
        self.all_contents.append(content)

    def add_contents(self, contents: List[Content]):
        self.all_contents += contents

    def replace_contents(self, contents: Union[List[dict], List[Content]]):
        try:
            if type(contents[0]) == dict:
                self.all_contents = [Content.from_dict(d) for d in contents]
            else:  # assuming type(contents[0]) == Content:
                self.all_contents = contents
        except:
            # Todo - maybe remove the sourcefile?
            print("[SKIPPING] Probably an empty file: {}".format(self.output_fp))

    def to_list_of_dicts(self):
        return [c.to_dict() for c in self.all_contents]

    def to_flat_list_of_dicts(self):
        return [self.flatten_nested_dict(c.to_dict()) for c in self.all_contents]

    def _flatten_dict_gen(self, content_dict, parent_key, sep):
        for k, v in content_dict.items():
            # new_key = parent_key + sep + k if parent_key else k
            new_key = k if parent_key else k
            if isinstance(v, collections.abc.MutableMapping):
                yield from self.flatten_nested_dict(v, new_key, sep=sep).items()
            else:
                yield new_key, v

    def flatten_nested_dict(self, content_dict: collections.abc.MutableMapping, parent_key: str = '', sep: str = '.'):
        return dict(self._flatten_dict_gen(content_dict, parent_key, sep))
