from typing import List, Optional, Dict

import os, glob, re
import logging
import subprocess
from tqdm import tqdm
from pathlib import Path

from customdocument import CustomDocument

logger = logging.getLogger(__name__)


def process_pdf_files_in_directory(input_directory: Path = "data/ir_data/pdf/",
                                   output_directory: Path = "data/ir_data/pdf_converted/"):
    """
    Convert all pdf files found in a directory (and subfolders) to CustomDocuments.
    """
    # make sure the output_directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # recursively look inside subfolders if they exist
    input_pdf_filepaths = [x for x in input_directory.glob("**/*.pdf")]
    for pdf_filepath in tqdm(input_pdf_filepaths):
        output_path = output_directory.joinpath(pdf_filepath.stem + ".json")
        if not output_path.exists():
            converted_document = convert_pdf_to_customdocument(pdf_filepath, output_path)
            converted_document.write_document()
        else:
            pass


def read_pdf(file_path: Path, layout: bool = True, encoding: Optional[str] = "Latin1") -> List[str]:
    """
    Extract pages from the pdf file at file_path; based on Haystack.

    :param file_path: path of the pdf file
    :param layout: whether to retain the original physical layout for a page. If disabled, PDF pages are read in
                   the content stream order.
    """
    if layout:
        command = ["pdftotext", "-enc", encoding, "-layout", str(file_path), "-"]
    else:
        command = ["pdftotext", "-enc", encoding, str(file_path), "-"]
    output = subprocess.run(command, stdout=subprocess.PIPE, shell=False)  # type: ignore
    document = output.stdout.decode(errors="ignore")
    pages = document.split("\f")
    pages = pages[:-1]  # the last page in the split is always empty.
    return pages


def convert_pdf_to_customdocument(
            source_file_path: Path,
            output_file_path: Path,
            remove_header_and_footer: Optional[bool] = True,
            clean_whitespace: Optional[bool] = True,
            clean_empty_lines: Optional[bool] = True,
            encoding: Optional[str] = "Latin1") -> CustomDocument:
    """
    Extract pages from the pdf file at file_path; based on Haystack.

    :param output_file_path:    Path to the .json file to store the converted file.
    :param source_file_path:    Path to the .pdf file you want to convert
    :param encoding: Encoding that will be passed as -enc parameter to pdftotext. "Latin 1" is the default encoding
                     of pdftotext. While this works well on many PDFs, it might be needed to switch to "UTF-8" or
                     others if your doc contains special characters (e.g. German Umlauts, Cyrillic characters ...).
                     Note: With "UTF-8" we experienced cases, where a simple "fi" gets wrongly parsed as
                     "xef\xac\x81c" (see test cases). That's why we keep "Latin 1" as default here.
                     (See list of available encodings by running `pdftotext -listenc` in the terminal)
    """
    pages = read_pdf(source_file_path, layout=True, encoding=encoding)

    if not pages:
        # empty input file
        return None

    pages = ["\n".join(p.splitlines()) for p in pages]

    # splitting text happens during preprocessing, so no split_size passed here;
    # split_size will be set to -1 during conversion.
    document = CustomDocument(output_file_path, source_file_path, split_size=-1)

    print("Converted PDF file to pages of text, combining to a single CustomDocument to keep track of page nrs.")
    for page_idx, page in tqdm(enumerate(pages)):

        # some simple cleaning -- roughly based on haystack.
        lines = page.splitlines()
        if remove_header_and_footer:
            # simplest way for removing header and footer
            lines = lines[1:-2]

        if clean_whitespace:
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                cleaned_lines.append(line)
            text = " ".join(cleaned_lines)

        if clean_empty_lines:
            text = re.sub(r"\n\n+", "\n\n", text)
            text = re.sub(r"[\s]+", " ", text)

        # no splitting here yet, so simply using page_nr as a place holder and split_id is left blank
        page_nr = str(page_idx + 1)
        document.add_content(text=text,
                             page_nr=page_nr,
                             doc_title=source_file_path.name)  # we're using the pdf file name for simplicity

    return document
