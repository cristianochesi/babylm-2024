import sys
import os
import re
import utils
from smart_open import open
import regex as rex

d_in = sys.argv[1]
d_out = sys.argv[2]
target_file = "gutenberg.txt"
text_original = ""
text_cleaned = ""
n_files = 0
utils.profiling_init(utils)
print("\n===GUTENBERG data processing===\nOpening files in '" + d_in + "':")

  
def preprocess(line):

    line = line.lower()
    line = utils.normalize_abbreviations(line)
    line = utils.handle_acronyms(line)
    line = utils.space_punctuation(line)
    line = utils.remove_quotes(line) #remove quotes
    line = utils.remove_symbols(line) #remove symbols
    line = utils.add_full_stop(line)
    line = re.sub(r'([\.,:;!?()—])', r' \1 ', line) # add space before and after punctuation (from utils.space_punctuation)
    line = re.sub(r'(\w)?([:;.!?…])(\w)?', r'\1\2\n\3', line) # split lines after strong punctuation (from utils.space_punctuation)
    line = re.sub(r'^[!\"#$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_\{\|\}~\s]+', '', line, flags=re.MULTILINE) #clean line beginning, ad hoc (more restrictive)
    line = re.sub(r'\s{2,}', ' ', line)  # remove multiple spacing (only one of the two present in utils.remove_multiple_spacing)

    return line


for file in os.listdir(d_in):
    print(".", end="")
    n_files += 1
    with open(d_in + file, encoding='utf8') as f:
        for line in f:
            text_original += line
            if not utils.is_empty_line(line):
                text_cleaned += preprocess(line)

utils.report(n_files, text_original, text_cleaned)
utils.save(d_out, target_file, text_original, text_cleaned)
utils.profiling_end(utils)