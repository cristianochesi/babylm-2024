import sys
import os
import re
import utils
from smart_open import open
from normalize import clean
import regex as rex

d_in = sys.argv[1]
d_out = sys.argv[2]
target_file = "subtitles.txt"
text_original = ""
text_cleaned = ""
n_files = 0
utils.profiling_init(utils)
print("\n===SUBTITLES data processing===\nOpening files in '" + d_in + "':")

def preprocess(line):
    if re.match(r'^[|=0123456789]', line) or utils.is_empty_line(line):
        return ""
    else:
        line = ' '.join(line.strip().split())
        line = clean(line, minimal=True)
        line = line.lower()
        line = utils.normalize_abbreviations(line)
        line = utils.handle_acronyms(line)
        line = re.sub(r'ã ','à', line)
        line = re.sub("ãƒâƒã'â", "à", line)
        line = re.sub("ãƒâƒã'â¨ãƒâƒã'â¨", "è", line)
        line = re.sub("<i>","",line)
        line = utils.remove_quotes(line)
        line = utils.remove_symbols(line)
        line = re.sub(r'(\w)?([;.!?…])(\w)?', r'\1\2\n\3', line) # split lines after strong punctuation, ad hoc, from from utils.space_punctuation (: removed from regex)
        line = re.sub(r'(\s)?([;.!?…])(\s)*', r'\1\2\n', line) #ad hoc, from utils.space_punctuation (: removed from regex)
        line = re.sub(r'([.,:;!?()\[\]])', r' \1', line)  # add space before punctuation (from utils.space_punctuation)
        line = re.sub(r'\. (\. )+', r' ... ', line)  # pauses (from utils.space_punctuation)
        line = re.sub(r'([,:()\[\]/])', r'\1 ', line)  #add space after these symbols, ad hoc (from utils.space_punctuation)
        line = rex.sub(r'\n[\p{P}\s]*', '\n', line) #replace newline followed by any punctuation or whitespace with a single newline
        line = utils.add_full_stop(line.strip())
        line = re.sub (r'^[!\"#$%&\'*+,-./:;<=>?@\^_{|}~\s]+','', line) #ad hoc, clean line beginning (no removal of () and [] at line beginning)
        line = utils.remove_multiple_spacing(line)
        return line+"\n"


for file in os.listdir(d_in):
    print(".", end="")
    n_files += 1
    with open(d_in + file, encoding='utf8') as f:
        prev_line = ""
        for line in f:
            text_original += line+"\n"
            if prev_line != line and len(line)>0:
                prev_line = line
                text_cleaned += preprocess(line)
            
utils.report(n_files, text_original, text_cleaned)
utils.save(d_out, target_file, text_original, text_cleaned)
utils.profiling_end(utils)