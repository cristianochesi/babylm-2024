import sys
import os
import re
import utils
from tqdm import tqdm
from smart_open import open
import ftfy
from line_profiler import LineProfiler

d_in = sys.argv[1]
d_out = sys.argv[2]
target_file = "wikipedia.txt"
text_original = ""
text_cleaned = ""
text_cleaned_lines = []
text_original_lines = []
n_files = 0
utils.profiling_init(utils)
print("\n===WIKIPEDIA data processing===\nOpening files in '" + d_in + "':")

regex_1 = re.compile(r"\[\d+\]") #footnotes
regex_2 = re.compile(r"\[\[([^\|\]])\|[^\]]*\]\]") #links
regex_3 = re.compile(r"= = = ([^\=]*) = = =") #section titles

def preprocess(line):  
    line = ' '.join(line.strip().split()) # Remove extra spaces and strip the line  
    line = ftfy.fix_text(line) # Clean the line (assuming clean is a custom function and can handle None)  
    line = line.lower()
    line = regex_1.sub("", line)
    line = regex_2.sub(r"\1", line)
    line = regex_3.sub(r"\1", line)
    line = utils.normalize_abbreviations(line)
    line = utils.handle_acronyms(line)
    line = utils.space_punctuation(line)
    line = utils.normalize_quotes(line)
    line = utils.space_symbols(line)
    line = utils.remove_brackets(line)
    line = utils.add_full_stop(line)
    line = utils.remove_multiple_spacing(line)
    return utils.remove_last_newline(line)+"\n"

profiler = LineProfiler()
profiler.add_function(preprocess)
profiler.enable()

for file in os.listdir(d_in):
    print(".", end="")
    n_files += 1
    with open(d_in + file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    with open(d_in + file, encoding='utf8') as f:
        for line in tqdm(f, total=total_lines, desc='Processing Lines', mininterval=60.0, miniters=10000):
            text_original_lines.append(line + "\n")
            if len(line) > 0 and not utils.is_empty_line(line):
                text_cleaned_lines.append(preprocess(line))

text_cleaned = ''.join(text_cleaned_lines)
text_original = ''.join(text_original_lines)
profiler.disable()
profiler.print_stats()
            
utils.report(n_files, text_original, text_cleaned)
utils.save(d_out, target_file, text_original, text_cleaned)
utils.profiling_end(utils)