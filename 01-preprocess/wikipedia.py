import sys
import os
import re
import utils
from smart_open import open
from normalize import clean
import regex as rex

d_in = sys.argv[1]
d_out = sys.argv[2]
target_file = "wikipedia.txt"
text_original = ""
text_cleaned = ""
n_files = 0
utils.profiling_init(utils)
print("\n===WIKIPEDIA data processing===\nOpening files in '" + d_in + "':")

regex_1 = re.compile(r"\[\d+\]") #footnotes
regex_2 = re.compile(r"\[\[([^\|\]])\|[^\]]*\]\]") #links
regex_3 = re.compile(r"= = = ([^\=]*) = = =") #section titles

def preprocess(line):  
    line = ' '.join(line.strip().split()) # Remove extra spaces and strip the line  
    line = clean(line, minimal=True) # Clean the line (assuming clean is a custom function and can handle None)  
    line = line.lower()
    
    line = regex_1.sub("", line)
    line = regex_2.sub(r"\1", line)
    line = regex_3.sub(r"\1", line)
    line = utils.normalize_abbreviations(line)
    line = utils.handle_acronyms(line)
    line = utils.normalize_quotes(line)
    line = utils.space_symbols(line)
    line = utils.add_full_stop(line)
    line = utils.space_punctuation(line) 
    line = utils.remove_multiple_spacing(line)
    line = utils.remove_empty_lines(line)
    line = re.sub(r"\( ;", ";", line)


    return line+"\n"

for file in os.listdir(d_in):
    utterances = ""
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