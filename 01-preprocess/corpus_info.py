import sys
import os
import utils
from smart_open import open

file = sys.argv[1]

print("\n===CLEANED CORPUS info===\nOpening file '" + file + "':")



with open(file, encoding='utf8') as f:
    text_original = f.read()

types, tokens = utils.get_corpus_info(text_original)
print("Types: " + str(types) + "\nTokens: " + str(tokens) + "\nTTR: " + str(round(types/tokens,2)))