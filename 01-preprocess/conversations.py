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
    line = clean(line, minimal=True)
    if utils.is_empty_line(line) or re.search(r'\|', line) or re.search('= = =', line) or re.search('<', line) or re.search('>', line):
        return ""
    else: 
        line = re.sub(r"\[(?!pausa\])[^]]*\]", "", line)
        line = re.sub(r"^[A-Z]:\s*", "", line)
        line = re.sub(r"a'", "à", line)
        line = re.sub(r"e'", "è", line)
        line = re.sub(r"i'", "ì", line)
        line = re.sub(r"o'", "ò", line)
        line = re.sub(r"u'", "ù", line)
        line = clean(line, minimal=True)
        line = line.lower()
        line = utils.remove_quotes(line)
        line = utils.remove_symbols(line)
        line = utils.space_punctuation(line)
        line = utils.add_full_stop(line.strip())
        line = utils.remove_multiple_spacing(line)
        if utils.is_empty_line(line):
            return ""
        else:
            return line+"\n"

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