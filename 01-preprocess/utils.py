import re
import os
import psutil
import time
import regex

global start_time, start_mem

PUNCTUATION_PATTERN = re.compile(r'([.,:;!?()—])')
MULTIPLE_DOTS_PATTERN = re.compile(r'\.( *\.)+')
SPLIT_LINES_PATTERN = re.compile(r'([:;.!?])\s*')
INITIAL_PUNCT_PATTERN = regex.compile(r'^[\p{P}\s]+')
NEWLINE_PUNCT_PATTERN = regex.compile(r'\n[\p{P}\s]+')
ADD_FULL_STOP = regex.compile(r'\p{P} *$')
SPACE_SYMBOLS = re.compile(r'([*+#@§&%$£])')
REMOVE_SYMBOLS = re.compile(r'([*+#@§&%$£_|=♪])')
REMOVE_BRACKETS = re.compile(r'[\[\]<>(){}]')
REMOVE_MULTI_DASH = re.compile(r'--+')
REMOVE_INITIAL_DASH = re.compile(r'^-*')
REMOVE_MULTI_SPACING = re.compile(r' +')
REMOVE_LAST_NEWLINE = re.compile(r'[\r\n]$')
IS_EMPTY = regex.compile(r'^[\s\d\p{P}]*$')
QUOTES = re.compile(r'[“‘’”"«»「」]')
ACRONYMS = re.compile(r'\b(?:[A-Z]{1,3}\.)+')
NUMBERS = re.compile(r'\b(\d+\.)+')
URL_PATTERN = re.compile(
    r'https?://(?:www\.)?'  # http:// or https:// followed by optional www.
    r'(?:[a-zA-Z0-9-]{1,63}\.)+'  # Domain name
    r'[a-zA-Z]{2,63}'  # Top-level domain
    r'(?:/\S*)?'  # Optional path and query parameters
)
ABBREVIATIONS_ENG = {
    r"mr\.": "mr", r"mrs\.": "mrs", r"ms\.": "ms", r"dr\.": "dr", r"drs\.": "drs",
    r"prof\.": "prof", r"st\.": "st", r"sr\.": "sr", r"jr\.": "jr", r"esq\.": "esq",
    r"m\.d\.": "md", r"ph\.d\.": "phd", r"etc\.": "etc", r"i\.e\.": "ie", r"e\.g\.": "eg",
    r"a\.m\.": "am", r"p\.m\.": "pm", r"b\.c\.": "bc", r"a\.d\.": "ad", r"cf\.": "cf",
    r"vs\.": "vs", r"ave\.": "ave", r"blvd\.": "blvd", r"rd\.": "rd", r"ln\.": "ln",
    r"no\.": "no", r"co\.": "co", r"inc\.": "inc", r"ltd\.": "ltd", r"et al\.": "et al",
    r"vol\.": "vol", r"pp\.": "pp", r"chap\.": "chap", r"fig\.": "fig", r"ed\.": "ed",
    r"n\.b\.": "nb", r"op\. cit\.": "op cit", r"loc\. cit\.": "loc cit", r"ibid\.": "ibid",
    r"id\.": "id", r"i\.q\.": "iq", r"o\.r\.": "or"
}

NORM_ABBR = re.compile(r'\b'.join(map(re.escape, ABBREVIATIONS_ENG.keys())))

def get_corpus_info(corpus):
    tokens = corpus.split()
    types = {}
    for t in tokens:
        if not t in types:
            types[t] = ""
    return len(types), len(tokens)

def space_punctuation(line):
    line = PUNCTUATION_PATTERN.sub(r' \1 ', line)  # Add space before and after punctuation, and handle pauses
    line = MULTIPLE_DOTS_PATTERN.sub(r' … ', line)
    line = SPLIT_LINES_PATTERN.sub(r'\1\n', line)  # Split lines after strong punctuation and ensure line breaks
    line = INITIAL_PUNCT_PATTERN.sub('', line)  # Remove initial useless punctuation and clean up newlines
    return line

def space_symbols(line):
    return SPACE_SYMBOLS.sub(r' \1 ', line)  # space symbols

def remove_symbols(line):
    return REMOVE_SYMBOLS.sub(' ', line)  # remove symbols

def remove_last_newline(line):
    return REMOVE_LAST_NEWLINE.sub('', line)  # remove symbols

def add_full_stop(line):
    if not ADD_FULL_STOP.search(line):
        line += ' .'
    return line

def remove_multiple_spacing(line):
    return REMOVE_MULTI_SPACING.sub(' ', line)  # remove multiple spacing

def remove_brackets(line):
    return REMOVE_BRACKETS.sub('', line)  # remove brackets

def normalize_quotes(line):
    return QUOTES.sub(' " ', line)

def remove_quotes(line):
    return QUOTES.sub('', line)

def remove_dashes(line):
    line = REMOVE_MULTI_DASH.sub('', line)  # remove multiple dashes
    line = REMOVE_INITIAL_DASH.sub('', line)  # remove initial dash(es)
    return line

def is_empty_line(line):
    if len(line) > 0 and not IS_EMPTY.match(line):
        return 0
    else:
        return 1

def remove_dots(match):
    return match.group().replace('.', '')

def handle_acronyms(line):
    line = ACRONYMS.sub(remove_dots, line)
    line = NUMBERS.sub(remove_dots, line)
    return line

def remove_links(line):
    return URL_PATTERN.sub('', line)  # remove url links

def normalize_abbreviations(text):
    return NORM_ABBR.sub(lambda match: ABBREVIATIONS_ENG[match.group(0)], text)

# reporting corpus information
def report(n_files, text_original, text_cleaned):
    print("\nNumber of files pre-processed: " + str(n_files))

    types, tokens = get_corpus_info(text_original)
    print("\nBefore cleaning:\nTypes: " + str(types) + "\nTokens: " + str(tokens) + "\nTTR: " + str(
        round(types / tokens, 2)))

    types, tokens = get_corpus_info(text_cleaned)
    print("\nAfter cleaning:\nTypes: " + str(types) + "\nTokens: " + str(tokens) + "\nTTR: " + str(
        round(types / tokens, 2)))

# Write original and cleaned text files
def save(d_out, target_file, text_original, text_cleaned):
    with open(d_out + "original/" + target_file, 'w', encoding='utf-8') as f:
        f.write(f"{text_original}")

    print("Original data stored in: " + d_out + "original/" + target_file)

    with open(d_out + target_file, 'w', encoding='utf-8') as f:
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
    print("Execution time: " + str(execution_time) + " min - Memory Load: " + str(memory_load) + " MB")