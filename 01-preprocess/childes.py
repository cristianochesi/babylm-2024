import sys
import os
import re
import utils
from smart_open import open
from normalize import clean

d_in = sys.argv[1]
d_out = sys.argv[2]
target_file = "childes.txt"
text_original_lines = []
text_cleaned_lines = []
n_files = 0
utils.profiling_init(utils)
print("\n===CHILDES data processing===\nOpening files in '" + d_in + "':")

def preprocess(line):
    line = ' '.join(line.strip().split())  # Remove extra spaces
    line = line[6:]                        # Remove the first 6 characters
    line = re.sub(r'&[\-\+]','', line)     # Remove patterns like &- or &+
    line = re.sub(r'\[.*?\]','', line)     # Remove anything inside square brackets
    line = re.sub('↫.*?↫', '', line)       # Remove text between '↫' symbols
    line = re.sub(r'\(\.+\)', ' , ', line) # Replace "(...)" with ","
    line = re.sub(r'[<>\\//]', '', line)   # Remove specific special characters
    line = clean(line, minimal=True)       # Clean again for any remnants
    line = line.lower()                    # Convert text to lowercase
    line = re.sub(r' è([a-z])', r' è \1', line) # space 'è' when wrongly attached to next word: "èquesto" > "è questo"
    line = utils.normalize_abbreviations(line)  # Normalize abbreviations
    line = utils.remove_quotes(line)            # Remove quotation marks
    line = utils.remove_symbols(line)           # Remove unwanted symbols
    line = utils.space_punctuation(line)        # Add space around punctuation
    line = utils.add_full_stop(line.strip())    # Add full stop at the end of the line
    line = utils.remove_multiple_spacing(line)  # Remove extra spaces
    
    # This removes lines that are just a period (with or without surrounding spaces)
    if re.match(r'^\s*\.\s*$', line):
        return ''  # Return an empty string to remove the line
    
    return line + "\n"  # Return the cleaned line with a newline character

for file in os.listdir(d_in):
    print(".", end="")
    n_files += 1
    with open(d_in + file, encoding='utf8') as f:
        for line in f:
            if not line.startswith(("[", "=", "@", "%", "*CHI")):
                text_original_lines.append(line)
                if not utils.is_empty_line(line):
                    text_cleaned_lines.append(preprocess(line))

text_original = ''.join(text_original_lines)
text_cleaned = ''.join(text_cleaned_lines)

utils.report(n_files, text_original, text_cleaned)
utils.save(d_out, target_file, text_original, text_cleaned)
utils.profiling_end(utils)
