import sys
import os
import re
import utils
from smart_open import open
from normalize import clean

d_in = sys.argv[1]
d_out = sys.argv[2]
target_file = "conversations.txt"
text_original = ""
text_cleaned = ""
n_files = 0
utils.profiling_init(utils)
print("\n===CONVERSATIONS data processing===\nOpening files in '" + d_in + "':")


def preprocess(line):
    line = ' '.join(line.strip().split())
    if line.startswith("- "):
        line = line[2:]
    elif line.startswith("-"):
        line = line[1:]
    elif line.startswith("A: "):
        line = line[3:]
    elif line.startswith("B: "):
        line = line[3:]

    line = clean(line, minimal=True)
    line = line.lower()
    line = utils.normalize_abbreviations(line) 
    line = utils.space_punctuation(line)
    line = utils.normalize_quotes(line)
    line = utils.space_symbols(line)
    line = utils.remove_brackets(line)
    line = utils.add_full_stop(line)
    line = utils.remove_multiple_spacing(line)
    return line+"\n"


for file in os.listdir(d_in):
    print(".", end="")
    n_files += 1
    with open(d_in + file, encoding='utf8') as f:
        for line in f:
            if len(line)>0:
                text_original += line
                if not utils.is_empty_line(line):
                    text_cleaned += preprocess(line)
            

utils.report(n_files, text_original, text_cleaned)
utils.save(d_out, target_file, text_original, text_cleaned)
utils.profiling_end(utils)