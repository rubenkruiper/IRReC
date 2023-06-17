import logging, os
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm
from utils import cleaning_utils
from utils.customdocument import CustomDocument

logger = logging.getLogger(__name__)


def process_html_files_in_directory(input_directory: Path = "data/ir_data/xml/",
                                    output_directory: Path = "data/ir_data/xml_converted/"):
    """

    """
    # make sure the output_directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    converted_documents = []
    # recursively look inside subfolders if they exist
    input_html_filepaths = [x for x in input_directory.glob("**/*.html")]
    for html_filepath in tqdm(input_html_filepaths):
        output_path = output_directory.joinpath(html_filepath.stem + ".json")
        if not output_path.exists():
            converted_document = convert_html_to_customdocument(html_filepath, output_path)
            converted_document.write_document()
            converted_documents.append(converted_document)
        else:
            converted_documents.append(CustomDocument.load_document(output_path))
    return converted_documents


def grab_html_text_simple(file_path: Path):
    """
    All text in the EU htmls seems to be captured neatly in <p> tags, we don't care about structure currently.
    We do remove all unicode characters, see `utils.remove_unicode_chars()`.
    """
    with open(file_path, 'r') as f:
        text = f.read()
    soup = BeautifulSoup(text, 'html.parser')
    return [cleaning_utils.remove_unicode_chars(x.text) for x in soup.body.find_all('p')]


def convert_html_to_customdocument(source_file_path: Path,
                          output_file_path: Path) -> CustomDocument:
    document = CustomDocument(output_file_path, source_file_path, split_size=-1)
    document_paragraphs = []
    list_of_paragraphs = grab_html_text_simple(source_file_path)
    for paragraph in list_of_paragraphs:
        if paragraph.strip() != '':
            document_paragraphs.append(paragraph)

    for paragraph_idx, paragraph in tqdm(enumerate(document_paragraphs)):
        # no splitting here yet, so simply using page_nr as a place holder and split_id is left blank
        paragraph_nr = str(paragraph_idx + 1)
        document.add_content(text=paragraph,
                             page_nr=paragraph_nr,
                             doc_title=source_file_path.name)  # we're using the html file name for simplicity
    return document
