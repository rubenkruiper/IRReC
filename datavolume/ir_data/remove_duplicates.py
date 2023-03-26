import json, glob, re, os
from pathlib import Path


def grab_doc_id(doc_fp: Path):
    """ 
    Grab a document identifier from its name, will be along the lines of BS 1234 2020.
    """

    document_name = doc_fp.stem
    
    match = None
    if document_name.startswith("BS") or document_name.startswith("NA") or document_name.startswith("+"):
        match = re.match(r"[A-Z +]+(to)?[A-Z ]+([\d\-+:]+)([A-Z ]+[\d\-+:]+)?", document_name)
    elif document_name[0].isdigit():
        match = re.match(r'[0-9: +]([\dA-Z\-+:]+)([A-Z ]+[\d\-+:]+)?', document_name)
    elif document_name.startswith("Eurocode"):
        match = re.match(r"Euro([\w :])+([\d\-+:]+)([A-Z ]+[\d\-+:]+)?", document_name)
    elif document_name.startswith("Approved"):
        match = re.match(r"Approved([\w ])+[-]", document_name)
    elif document_name.startswith("LPS"):
        match = re.match(r"LPS([\w ])+[-]", document_name)
    elif document_name.startswith("DD"):
        match = re.match(r"DD[CENVTS -13678]+[210]+", document_name)
    elif document_name.startswith(("CP")):
        match = re.match(r"[CPa-z ]+([\d -])+", document_name)
    elif document_name.startswith(("CFPA")):
        match = re.match(r"CFPA([A-Za-z _]+[\d -_])+", document_name)
    elif document_name.startswith(("NT")):
        match = re.match(r"NT([Bbuild -]+)[\d -]+", document_name)
    elif document_name.startswith(("PD")):
        match = re.match(r"PD[A-Z ]+[\d -]+", document_name)
    elif document_name.startswith(("PAS")):
        match = re.match(r"PAS[\d -]+", document_name)
        
    if match:
        end_idx = match.end()
        document_name = document_name[:end_idx]
        if "+A" in document_name:
            document_name = document_name.split("+A", 1)[0]
        while document_name.endswith(("-", "-", " ",":", "+")):
            document_name = document_name[:-1]
    
    return document_name


def check_for_tracked_changes(doc_fp: Path):
    """ Check whether a document is a tracked changes doc, we don't want those! """
    document_name = doc_fp.stem
    tracked_changes_indicators = ["TC", "Tracked", "tracked"]
    for tci in tracked_changes_indicators:
        if tci in document_name:
            return "tracked changes"
    return ""


# ---- script

input_path = Path("foreground_pdf")
pdf_filepaths = [x for x in input_path.glob("**/*.pdf")]

file_tracker_dict = {}
for pdf_fp in pdf_filepaths:
    file_tracker_dict[str(pdf_fp)] = {
        "location": pdf_fp.parent,
        "name": pdf_fp.stem,
        "doc_id": grab_doc_id(pdf_fp),
        "tracked changes": check_for_tracked_changes(pdf_fp)
    }

unique_doc_ids = set(fp_info["doc_id"] for fp_info in file_tracker_dict.values() if not fp_info['tracked changes'])

# remove the files that have a duplicate with year at the end
duplicate_year_only = []
for udi in unique_doc_ids:
    if udi[:-5] in unique_doc_ids:
        duplicate_year_only.append(udi)

for dyo in duplicate_year_only:
    unique_doc_ids.remove(dyo)
    
unique_fps_to_keep = []
for fp, fp_info in file_tracker_dict.items():
    if fp_info["doc_id"] in unique_doc_ids:
        unique_fps_to_keep.append(fp)
        unique_doc_ids.remove(fp_info["doc_id"])

# simply remove all the duplicate files.
number_removed_files = 0
for fp in pdf_filepaths:
    if str(fp) not in unique_fps_to_keep:
        os.remove(fp)
        number_removed_files += 1

print(f"Number of files before removal: {len(pdf_filepaths)}")
print(f"Number of files AFTER removal: {len(unique_doc_ids)}")
print(f"Number of files removed: {number_removed_files} should equal {len(pdf_filepaths) - len(unique_doc_ids)}")