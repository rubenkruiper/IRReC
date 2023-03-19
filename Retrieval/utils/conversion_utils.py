from typing import List
from pathlib import Path
from utils.customdocument import CustomDocument
from utils.converter_pdf import process_pdf_files_in_directory
from utils.converter_html import process_html_files_in_directory
# from Retrieval.utils.converter_xml import process_xml_files_in_directory


def convert_inputs(conversion_input_dir: Path,
                   conversion_output_dir: Path,
                   conversion_type: str) -> List[CustomDocument]:
    """
    Converts documents to CustomDocument objects with the desired per-page dict/json format for preprocessing.
    """
    if conversion_type.startswith("pdf"):
        process_pdf_files_in_directory(conversion_input_dir, conversion_output_dir)
    elif conversion_type.startswith("html"):
        process_html_files_in_directory(conversion_input_dir, conversion_output_dir)
    # elif self.conversion.startswith("xml"): # todo; need to update the xml converter
    #     converted_documents = process_xml_files_in_directory(self.input_dir, self.converted_output_dir)
    else:
        raise ValueError("No valid input type provided to convert_inputs()")
