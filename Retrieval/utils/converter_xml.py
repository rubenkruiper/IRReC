from typing import List, Optional, Dict, Any, Generator, Set
from functools import partial, reduce
from itertools import chain

import os, glob, json
import logging
import subprocess
from pathlib import Path

from bs4 import BeautifulSoup
from custom_haystack_nodes.other.customdocument import CustomDocument
from haystack.nodes.file_converter import BaseConverter
from textblob import TextBlob

from transformers import BertTokenizer


logger = logging.getLogger(__name__)


## TODO ---------------------------------------------------
##  This is currently not updated to the new approach to converting documents!
##  Need to compare against converter_html and update this code to process the XML regulations we have.
## TODO ---------------------------------------------------


class XMLToPageTextConverter(BaseConverter):
    def __init__(
            self,
            bert_model: Optional[str] = 'bert-base-cased',
            cache_dir: Optional[Path] = None,
            remove_numeric_tables: bool = False,
            valid_languages: Optional[List[str]] = None,
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        if cache_dir:
            # store pretrained embeddings locally, so an internet connection isn't required every time for loading them
            self.tokenizer = BertTokenizer.from_pretrained(bert_model, cache_dir=cache_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model)

        # save init parameters to enable export of component config as YAML
        self.set_config(
            remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages
        )

        verify_installation = subprocess.run(["pdftotext -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception(
                """pdftotext is not installed. It is part of xpdf or poppler-utils software suite.

                   Installation on Linux:
                   wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.03.tar.gz &&
                   tar -xvf xpdf-tools-linux-4.03.tar.gz && sudo cp xpdf-tools-linux-4.03/bin64/pdftotext /usr/local/bin

                   Installation on MacOS:
                   brew install xpdf

                   You can find more details here: https://www.xpdfreader.com
                """
            )

        super().__init__(
            remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages
        )

    def process_xml_files_in_directory(self,
                                       input_directory: Path = "data/ir_data/pdf/",
                                       output_directory: Path = "data/ir_data/pdf_converted/"):

        # make sure the output_directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        xml_files = glob.glob(input_directory + "**/*.xml", recursive=True)
        # recursively look inside subfolders
        processed_files = set([fp.rsplit('/', 1)[1] for fp in glob.glob(output_directory + "*.json")])

        for xml_fp in xml_files:
            doc_name = xml_fp.rsplit('/', 1)[1][:-3] + 'json'
            output_file_path = output_directory + doc_name
            if doc_name in processed_files and open(output_file_path, 'r').read():
                # Converted before, assuming the document was 100% processed
                print("[Converter] Already converted: {}".format(doc_name))   # split_size will be set to -1

            else:
                # convert and save contents to a file
                print("[Converter] Converting: {}".format(doc_name))
                converted_document = self.convert(output_file_path, xml_fp)
                if converted_document and any([c.text for c in converted_document.all_contents]):
                    converted_document.write_document()
                else:
                    print("[Converter] Issue converting the file at: {}".format(xml_fp))

    def convert(
            self,
            output_file_path: Path,
            source_file_path: Path,
            meta: Optional[Dict[str, str]] = None,
            remove_numeric_tables: Optional[bool] = False,
            clean_header_footer: Optional[bool] = True,
            encoding: Optional[str] = "ascii",
    ) -> CustomDocument:
        """
        Extract text from a .pdf file using the pdftotext library (https://www.xpdfreader.com/pdftotext-man.html)

        :param output_file_path:    Path to the .json file to store the converted file.
        :param source_file_path:    Path to the .pdf file you want to convert
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
                     Can be any custom keys and values.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param encoding: Encoding that will be passed as -enc parameter to pdftotext. "Latin 1" is the default encoding
                         of pdftotext. While this works well on many PDFs, it might be needed to switch to "UTF-8" or
                         others if your doc contains special characters (e.g. German Umlauts, Cyrillic characters ...).
                         Note: With "UTF-8" we experienced cases, where a simple "fi" gets wrongly parsed as
                         "xef\xac\x81c" (see test cases). That's why we keep "Latin 1" as default here.
                         (See list of available encodings by running `pdftotext -listenc` in the terminal)
        """
        # doc_name = file_path.rsplit('/', 1)[1]
        std_ref, title, xml_text = self._read_xml(source_file_path)
        doc_title = f"{std_ref} # {title}"

        if not xml_text:
            # empty input file
            return None

        # splitting text happens during preprocessing, so no split_size passed here
        # split_size will be set to -1 during conversion
        document = CustomDocument(output_file_path, source_file_path, split_size=-1)
        document.add_content(text=xml_text, page_nr=0, doc_title=doc_title)

        return document

    def _read_xml(
            self, file_path: Path, encoding: Optional[str] = "ascii"
    ) -> List[str]:
        """
        Extract pages from the pdf file at file_path.

        :param file_path: path of the pdf file
        :param layout: whether to retain the original physical layout for a page. If disabled, PDF pages are read in
                       the content stream order.
        """
        soup = BeautifulSoup(open(file_path, "r").read(), 'xml')
        title = soup.find("title-wrap", {"xml:lang": 'en'}).get_text()
        std_ref = soup.find("std-ref", {"type": "dated"}).get_text()

        all_texts = []
        for tag in soup.findAll(["sec", "ref-list"]):
            text = tag.get_text().encode("ascii", errors="ignore").decode()
            # make sure we don't grab sections twice for nested sections...
            text_exists = bool([existing_text for existing_text in all_texts if text in existing_text])
            if not text_exists:
                all_texts.append(text)

        xml_text = ' '.join(all_texts)

        return std_ref, title, xml_text


