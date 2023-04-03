from typing import List, Union
import re
import copy

from Levenshtein import distance as lev


def split_list(some_list: List, chunk_size: int) -> List[List]:
    """
    Helper function to split a list into smaller lists of a given size.

    :param some_list:   List that has to be split into chunks.
    :param chunk_size:  Size of the sublists that will be returned.
    :return list_of_sublists:  A list of sublists, each with a maximum size of `chunk_size`.
    """
    return [some_list[i:i + chunk_size] for i in range(0, len(some_list), chunk_size)]


def repeating(span):
    try:
        parts = span.strip().split(' ').strip()
        if len(parts) > 1 and parts[0].lower() == parts[1].lower():
            return parts[0]
        else:
            return False
    except:
        return False


def custom_cleaning_rules(objects: Union[List[str], str]):
    """
    objects can be a List[str] or str
    """
    input_type = 'list'
    if type(objects) == str:
        input_type = 'str'
        objects = [objects]

    cleaned_objects = []
    for obj in objects:
        # remove double determiners that are sometimes grabbed, and strip objects
        obj = obj.replace("thethe", '', 1).strip()
        obj = obj.replace("thenthe", '', 1).strip()
        obj = obj.replace("thethat", '', 1).strip()
        obj = obj.replace("their ", '', 1).strip()
        # remove some unwanted types of punctuation
        obj = obj.replace(". ", '').strip()
        obj = obj.replace(" .", '').strip()
        obj = obj.replace("'", '').strip()
        obj = obj.replace('"', '').strip()
        obj = obj.replace("`", '').strip()
        while obj.startswith((":", ";", "-", "[", "(", ")", "]", "{", "}", "/", "\\", "$", "&", "@")):
            obj = obj[1:].strip()
        while obj.endswith((":", ";", "-", "[", "(", ")", "]", "{", "}", "/", "\\", "$", "&", "@")):
            obj = obj[:-1].strip()

        if repeating(obj):
            # remove instance of duplicate terms (e.g. from tables something like `floor floor')
            obj = repeating(obj)

        if len(obj) == 1:
            # remove 1 character objects
            continue
        elif len(obj) < 4 and (not obj.isupper() or any(c for c in obj if (c.isdigit() or c.isspace()))):
            # remove 2 characters objects that aren't all uppercase (abbreviations?) / contain a number or space
            # while removing some 3 letter words like 'ice' and 'fan', most of these are uninformative/erroneous
            continue
        elif len(obj) < 6 and len(re.findall(r"[^\w\s]", obj)) > 1:

            # any span of 5 characters or less, that contains multiple non-word and non-space characters
            continue
        elif len(re.findall(r"[=+*@|<>»_%]", obj)) > 0:
            # any span that may indicate that its taken from an equation or email address or simply gibberish from ocr
            continue
        elif obj.startswith("the ") or obj.startswith("The ") or obj.startswith("a ") or obj.startswith("A "):
            # do the same 1 char and 2/3 char removal in case the object starts with a determiner;
            if len(obj) == 5:
                continue
            elif len(obj) < 8 and obj[4:].islower():
                continue
            else:
                cleaned_objects.append(obj)
        else:
            cleaned_objects.append(obj)

    if input_type == 'list':
        return list(set(cleaned_objects))
    if input_type == 'str':
        try:
            return cleaned_objects[0]
        except IndexError:
            return ''


def remove_unicode_chars(text):
    text = text.replace("\u00a0", ' ')  # no-break-space
    text = text.replace("\u00a3", 'pounds ')  # £
    text = text.replace("\u00b2", '#SUP#2#SUP#')  # superscript 2
    text = text.replace("\u00b3", '#SUP#3#SUP#')  # superscript 2
    text = text.replace("\u00b0", ' degrees ')  # degrees sign
    text = text.replace("\u00ba", ' degrees ')  # degrees sign ~ masculine ordinal coordinator
    text = text.replace("\u00bd", '1/2')  # vulgar fraction half
    text = text.replace("\u00be", '3/4')  # vulgar fraction quarter
    text = text.replace("\u03bb", 'lambda')  # λ lambda
    text = text.replace("\u00e9", 'e')  # é
    text = text.replace("\u2013", '-')  # en-dash
    text = text.replace("\u2014", '-')  # em-dash
    text = text.replace("\xe2", '-')  # dash
    text = text.replace("\u2018", '`')  # left-single quotation mark
    text = text.replace("\u201c", '``')  # left-double quotation mark
    text = text.replace("\u2019", "'")  # right-single quotation mark
    text = text.replace("\u201d", "''")  # right-double quotation mark
    text = text.replace("\u2026", "...")  # horizontal ellipses
    text = text.replace("\uf059", "PSI")  # psi sign
    text = text.replace("\u00f7", "/")  # psi sign
    text = text.replace("\u2028", '\n')  # line separator
    text = text.replace("\xa0", " ")  # space
    text = text.replace("\xe2\x96\xba", "")  # arrow right
    text = text.replace("\xe2\x97\x84", "")  # arrow left
    text = text.replace("\xe2\x80\xa2", "")  # bullet point
    return text


def remove_determiners(text):
    dets = ["The ", "the ", "a ", "an ", "A ", "An ", "This ", "this ", "These ", "these ", "That ", "that ", "Those ",
            "those "]
    for d in dets:
        while text.startswith(d):
            text = text.replace(d, '', 1)
        if text.strip() == d.strip():
            text = ""
    return text


def levenshtein(w1: str, w2: str) -> bool:
    """
    Determine the Levenshtein distance between two spans, divided by the length of the longest span. If this
    value is below a given threshold (currently hardcoded to .75) the spans are considered dissimilar. This
    function is used to retrieve 'similar spans' from a cluster, with the aim to return spans that aren't close
    too similar to the given span; e.g., if the span is `test` the aim is to not return `tests` or `Test`.
    :param w1:  First of two spans to compare.
    :param w2:  Second of two spans to compare.
    :return bool:   Returns True for words that share a lot of characters, mediated by the length of the
                    longest input.
     """
    #   More character overlap --> smaller levenshtein distance
    # remove determiner
    if w1.startswith('the ') or w1.startswith('a ') or w1.startswith('The ') or w1.startswith('A '):
        w1 = w1.split(' ', 1)[1]
    if w2.startswith('the ') or w2.startswith('a ') or w2.startswith('The ') or w2.startswith('A '):
        w2 = w2.split(' ', 1)[1]

    if len(w1) > len(w2):
        long_w, short_w = w1, w2
    else:
        short_w, long_w = w1, w2

    # Comparison is between lowercased words, in order to ignore case
    # % 75% character level similarity minimum
    return 100 - (lev(short_w.lower(), long_w.lower()) / len(long_w) * 100) > 75


def cleaning_helper(to_be_cleaned: List[str]):
    """
    Helper function to call the basic filtering steps outlined in the cleaning utilities script.
    """
    basic_cleaned = custom_cleaning_rules(to_be_cleaned)
    determiners_removed = [remove_determiners(t) for t in basic_cleaned]
    cleaned_terms = [t for t in determiners_removed if t]
    return cleaned_terms
