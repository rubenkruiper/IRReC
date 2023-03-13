from typing import List
import re
import copy


class RegexFilter:
    def __init__(self):
        pass

    def single_regex_pattern(self, some_pattern, texts):
        """
        Helper function that applies a single regex pattern to a list of textspans.
        """
        no_updated = 0
        p = re.compile(some_pattern)
        new_texts = texts
        removed = []
        for idx, t in enumerate(texts):
            # check that the object wasn't already removed in a previous pass
            if t != '':
                t_ = re.sub(p, '', t)
                new_texts[idx] = t_

                if not t_:
                    # empty str after re.sub
                    removed.append(t)
                elif t_ != t:
                    # different str after re.sub
                    no_updated += 1

        # print("Removed {} objects, updated {}".format(len(removed), no_updated))
        return new_texts, removed

    def run_filter(self, to_be_filtered, regex_dict=None):
        """
        Function that can be called to run a set of regular expressions to filter out specific spans or parts of spans.
        Basic regexes are provided for identifying title_numbers, references and gibberish numbers in text.
        """
        if not regex_dict:
            # todo: improve regexes for preprocess-filtering
            regex_dict = {
                'title_numbers': '^([A-Z]{1}[. ]{1})?([ \d.+-])*',  # (?<![: ])(?![\D\-:]{1})
                'references': '[(]+([\d\s.])*[)]?',
                'gibberish_numbers': '^(\d|\w|_|—|@|=|\/|\\|~|\.|,|<|>|:|°|\*|\||\(|\))(?(1)(\s?(\d|_|—|@|=|\/|\\|~|\.|,|<|>|:|%|\*|\||\(|\))\s?)+(\w(?!\w))?|)',
                #     'real_numbers': '^\d*((\s)?(.|,)?(\s)?\d)*$',
            }

        removed_objects = []
        if type(to_be_filtered) == str:
            to_be_filtered = [to_be_filtered]

        updated_objects = copy.deepcopy(to_be_filtered)

        for filter_type, pattern in regex_dict.items():
            # print("Filtering {}".format(filter_type))
            updated_objects, removed = self.single_regex_pattern(pattern, updated_objects)
            removed_objects += removed

        if '' in updated_objects:
            updated_objects.remove('')

        return removed_objects, updated_objects


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
        parts = span.strip().lower().split(' ').strip()
        if len(parts) > 1 and parts[0] == parts[1]:
            return parts[0]
        else:
            return False
    except:
        return False


def custom_cleaning_rules(objects):
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
