import re
import os
import psutil
import time
import regex as rex

global start_time, start_mem

def get_corpus_info(corpus):
    tokens = corpus.split()
    types = {}
    for t in tokens:
        if not t in types:
            types[t] = ""
    return len(types), len(tokens)
     
def space_punctuation(line):  
    line = re.sub(r'([\.,:;!?()—])', r' \1 ', line) # add space before and after punctuation
    line = re.sub(r'\. (\. )+', r' ... ', line)  # pauses
    line = re.sub(r'(\w)?([:;.!?…])(\w)?', r'\1\2\n\3', line) # split lines after strong punctuation
    line = re.sub(r'(\s)?([:;.!?…])(\s)*', r'\1\2\n', line) #ensures presence of line break after punctuation    
    line = rex.sub(r'^[\p{P}\s]*', '', line.strip()) # remove initial useless punctuation
    line = rex.sub(r'\n[\p{P}\s]*', '\n', line) # Replace newline followed by any punctuation or whitespace with a single newline
    return line

def space_symbols(line): 
    line = re.sub(r'([*+#@§&%$£°])', r' \1 ', line)  # space symbols
    return line
    
def remove_symbols(line): 
    line = re.sub(r'([*+\-#@§&%$£°_\|=♪])', ' ', line)  #remove symbols
    return line

def add_full_stop(line):   
    if not rex.search(r'\p{P}$', line):
        line += ' . '
    return line

def remove_multiple_spacing(line): 
    line = re.sub(r'\s{2,}', ' ', line) # remove multiple spacing
    line = re.sub(r'\n{2,}', '\n', line) # remove multiple breaks
    return line
    
def remove_brackets(line): 
    line = re.sub(r'[\[\]<>\(\){}]', '', line) # remove brackets
    line = re.sub(r'--*', '', line) # remove multiple dashes
    line = re.sub(r'^- *', '', line) # remove initial dash
    return line
    
def normalize_quotes(line): 
    line = re.sub(r'[“‘’”""«»「」]', r' " ', line)
    line = re.sub(r'[‘’]', r'\'', line)
    return line
    
def remove_quotes(line): 
    line = re.sub(r'[“”""«»「」]', r'', line)
    return line

def is_empty_line(line):
    if len(line)>0 and line != "\n" and not rex.match(r'^[\s\p{P}]*$', line):
        return 0
    else:
        return 1

def remove_empty_lines(text):
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if not is_empty_line(line)]
    return '\n'.join(non_empty_lines)

def replace_dots(match):
        return match.group(0).replace('.', '')
    
def handle_acronyms(line):
    line = re.sub((r'\b(?:[a-z]\.)+(?:[a-z]{2,})?'), replace_dots, line) #Substitutes dots within acronyms
    line = re.sub(r'(\d+)\.(\d+)', r'\1\2', line) #Substitutes dots within numbers
    line = re.sub(r'\.[a-zA-Z]+\.', '', line) #substitutes patterns like .word.word2
    line = re.sub(r'\.[a-zA-Z]+', lambda m: '' + m.group(0)[1:], line) #handles the domains
    return line

def clean_line_beginning(text):
    pattern = r'^[!\"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~\s]+'

    while True:
        lines = text.splitlines()
        cleaned_lines = [re.sub(pattern, '', line) for line in lines]
        if all(not re.match(pattern, line) for line in cleaned_lines):
            break
        text = '\n'.join(cleaned_lines)
    return text

def normalize_abbreviations(text):
    text = text.replace("mr.", "mr")
    text = text.replace("mrs.", "mrs")
    text = text.replace("ms.", "ms")
    text = text.replace("dr.", "dr")
    text = text.replace("drs.", "drs")
    text = text.replace("prof.", "prof")
    text = text.replace(" st.", " st")
    text = text.replace(" sr.", " sr")
    text = text.replace(" jr.", " jr")
    text = text.replace("esq.", "esq")
    text = text.replace("m.d.", "md")
    text = text.replace("ph.d.", "phd")
    
    text = text.replace("etc.", "etc")
    text = text.replace("i.e.", "ie")
    text = text.replace("e.g.", "eg")
    text = text.replace("a.m.", "am")
    text = text.replace("p.m.", "pm")
    text = text.replace("b.c.", "bc")
    text = text.replace("a.d.", "ad")
    text = text.replace("cf.", "cf")
    text = text.replace("vs.", "vs")
    
    text = text.replace("ave.", "ave")
    text = text.replace("blvd.", "blvd")
    text = text.replace("rd.", "rd")
    text = text.replace("ln.", "ln")
    text = text.replace(" no.", " no")
    text = text.replace(" co.", " co")
    text = text.replace(" inc.", " inc")
    text = text.replace(" ltd.", " ltd")
    text = text.replace(" et al.", " et al")
    
    text = text.replace(" vol.", " vol")
    text = text.replace(" pp.", " pp")
    text = text.replace(" chap.", " chap")
    text = text.replace(" fig.", " fig")
    text = text.replace(" ed.", " ed")
    text = text.replace(" n.b.", " nb")
    text = text.replace(" op. cit.", " op cit")
    text = text.replace(" loc. cit.", " loc cit")
    text = text.replace("ibid.", "ibid")
    text = text.replace(" id.", " id")
    text = text.replace("i.q.", "iq")
    text = text.replace(" o.r.", " or")
    
    return text

# reporting corpus information
def report(n_files, text_original, text_cleaned):
    print("\nNumber of files pre-processed: " + str(n_files))

    types, tokens = get_corpus_info(text_original)
    print("\nBefore cleaning:\nTypes: " + str(types) + "\nTokens: " + str(tokens) + "\nTTR: " + str(round(types/tokens,2)))
    
    types, tokens = get_corpus_info(text_cleaned)
    print("\nAfter cleaning:\nTypes: " + str(types) + "\nTokens: " + str(tokens) + "\nTTR: " + str(round(types/tokens,2)))

# Write original and cleaned text files
def save(d_out, target_file, text_original, text_cleaned):
    with open(d_out+"original/"+target_file, 'w', encoding='utf-8') as f:
        f.write(f"{text_original}")
    
    print("Original data stored in: " + d_out + "original/" + target_file)
    
    with open(d_out+target_file, 'w', encoding='utf-8') as f:
        f.write(f"{text_cleaned}")
    print("Pre-processed data stored in: " + d_out + target_file)

def measure_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return round(mem_info.rss / (1024 ** 2), 2)  # Convert bytes to MB

def profiling_init(self):
    self.start_time = time.time()
    self.start_mem = measure_memory()

def profiling_end(self):
    execution_time = round((time.time() - self.start_time) / 60, 0)
    memory_load = measure_memory() - self.start_mem
    print("Execution time: "+str(execution_time)+" min - Memory Load: "+str(memory_load)+" MB")
