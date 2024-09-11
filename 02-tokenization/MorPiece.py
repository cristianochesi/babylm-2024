import os
import json
from math import log


class MorPiece:
    def __init__(self, vocab_size=30000, min_frequency=2, cutoff=8, bf=10, special_tokens=None):
        self.tokenization_to_print = "TP left-right \t BF right-left \t TP right-left \t BP right-left\n" # for debugging only
        if special_tokens is None:
            special_tokens = ['[unk]', '[pad]', '[sos]', '[eos]', '[pause]']
        self.special_tokens = special_tokens
        self.reserved_keys = {'[RSX]', '##', 'IDX', '++'}
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.bf = bf
        self.roots = {'[RSX]': {}, '++': {}}
        self.roots_unoptimized = {}
        self.infls = {}
        self.types = {}
        self.last_item_in_trie = {}
        self.idx = 0
        self.tokens = []
        self.suffixes = []
        self.tokens_bf = []
        self.suffixes_bf = []
        self.prefix = ""
        self.n_prefix = 0
        self.n_suffix = 0
        self.tokenized_words = []
        self.tokenized_word_longest = ""
        self.tokenized_word_idx_longest = ""
        self.cutoff = cutoff  # ln(8) is > 2, so, non-branching paths will be ignored
        self.num_tokens_in_corpus = 0
        self.num_chars_in_corpus = 0
        self.num_chars_in_trie = 0
        self.num_chars_in_optimized_trie = 0
        self.set_special_tokens(self.special_tokens)

    def train(self, corpus):  # create the vocabulary
        words = corpus.split()
        for word in words:
            word_alpha = ''.join([char for char in word if char.isalpha() or char == "'"])
            if not word_alpha:
                word = ''.join([char for char in word])
            else:
                word = word_alpha
            if word:
                self.build_trie(word, self.roots_unoptimized)  # create roots trie
                self.build_trie(word[::-1], self.infls)  # create inflections trie
                if word not in self.types:  # count tokens and chars in corpus
                    self.types[word] = 1
                else:
                    self.types[word] += 1
                self.num_tokens_in_corpus += 1
                self.num_chars_in_corpus += len(word)
        self.types = dict(sorted(self.types.items(), key=lambda item: item[1], reverse=True))

        sort_trie_by_freq(self.roots_unoptimized)
        sort_trie_by_freq(self.infls)
        self.optimize(self.types)

    def build_trie(self, wordpiece, root):  # build the trie and register # of traversals in '##'
        if wordpiece[0] in root:
            root[wordpiece[0]]['##'] += 1
            self.num_chars_in_trie += 1
            if len(wordpiece) > 1:
                self.build_trie(wordpiece[1:], root[wordpiece[0]])
            else:
                if 'END' not in root[wordpiece[0]]:
                    root[wordpiece[0]]['END'] = None
        else:
            root[wordpiece[0]] = {}
            root[wordpiece[0]]['##'] = 1
            if len(wordpiece) > 1:
                self.build_trie(wordpiece[1:], root[wordpiece[0]])

    def set_special_tokens(self, list):
        for item in list:
            if item not in self.roots['[RSX]'].keys():
                self.roots['[RSX]'][item] = {'IDX': None}
                self.roots['[RSX]'][item]['IDX'] = self.idx
                self.idx += 1

    # assign idx based on word freq and add potential inflection links in the root trie, remove frequency at the end
    def optimize(self, words):
        for word in words.keys():
            self.tokens = []
            self.suffixes = []
            self.tokens_bf = []
            self.suffixes_bf = []
            self.tokens.append(word[0])
            self.suffixes.append(word[len(word)-1])
            self.split_prefix(word, self.roots_unoptimized)
            if len(self.tokens) > 1:
                self.split_suffix(word[::-1], self.infls)
                self.suffixes = [word[::-1] for word in self.suffixes][::-1]
                self.tokenization_to_print += str(self.tokens) + '\t' + str(self.tokens_bf) + '\t'+ str(self.suffixes) + '\t' + str(self.suffixes_bf) + '\n'# for debugging only
                for i in range(0, len(self.tokens)): # esperimenti: usare solo self.suffixes o self.tokens (prefissi)
                    if i == 0:
                        self.last_item_in_trie = self.roots
                        self.add_items_to_trie(self.tokens[0]) # esperimenti: usare solo self.suffixes o self.tokens (prefissi)
                    else:
                        self.last_item_in_trie = self.roots['++']
                        self.add_items_to_trie(self.tokens[i]) # esperimenti: usare solo self.suffixes o self.tokens (prefissi)
                    if 'IDX' not in self.last_item_in_trie:
                        self.last_item_in_trie['IDX'] = self.idx
                        self.idx += 1
            else:
                self.last_item_in_trie = self.roots
                self.add_items_to_trie(word)
                if 'IDX' not in self.last_item_in_trie:
                    self.last_item_in_trie['IDX'] = self.idx
                    self.idx += 1

    def encode(self, sentence):
        self.tokenized_words = []
        words = sentence.split()
        for word in words:
            if word in self.roots['[RSX]']:
                self.tokenized_words.append([word, self.roots['[RSX]'][word]['IDX']])
            else:
                self.tokenized_word_longest = ""
                self.tokenized_word_idx_longest = ""
                self.retrieve(word, self.roots)
        return [sublist[1] for sublist in self.tokenized_words]

    def decode(self, sentence_idxs):
        tokens = []
        for idx in sentence_idxs:
            keys_path = find_idx_path(self.roots, idx)
            if keys_path:
                token = "".join(keys_path)
                if token.startswith('[RSX]'):
                    token = token[5:]
                tokens.append(token)
        return tokens

    def retrieve(self, word, trie):
        self.longest_match_in_trie(word, trie)
        if self.tokenized_word_longest:
            self.tokenized_words.append([self.tokenized_word_longest, self.tokenized_word_idx_longest])
        else:
            self.tokenized_words.append(['[unk]', self.roots['[RSX]']['[unk]']['IDX']])

    def longest_match_in_trie(self, string, trie):
        if string[0] in trie:
            self.tokenized_word_longest += string[0]
            if 'IDX' in trie[string[0]]:
                self.tokenized_word_idx_longest = trie[string[0]]['IDX']
            if len(string) > 1:
                self.longest_match_in_trie(string[1:], trie[string[0]])
        else:
            # print(string[0], self.tokenized_word_longest)
            if string[0] in self.roots['++'] and self.tokenized_word_idx_longest:
                self.tokenized_words.append([self.tokenized_word_longest + '++', self.tokenized_word_idx_longest])
                self.tokenized_word_longest = '++'
                self.tokenized_word_idx_longest = ''
                self.longest_match_in_trie(string, self.roots['++'])
            else:
                self.tokenized_words.append(['[unk]', self.roots['[RSX]']['[unk]']['IDX']])
                self.tokenized_word_longest = None

    def split_prefix(self, word, trie):
        l = len(word)
        if l > 1:
            self.get_pair_in_trie(word[0], word[1], trie)
            if self.check_tp(self.n_prefix, self.n_suffix) and self.get_bf(trie[word[0]]) <= self.bf:
                self.tokens.append(word[1])
                self.tokens_bf.append(word[0] + str(self.get_bf(trie[word[0]])))
            else:
                self.tokens[len(self.tokens) - 1] = self.tokens[len(self.tokens) - 1] + word[1]
        if l > 2:
            self.split_prefix(word[1:], trie[word[0]])

    def split_suffix(self, word, trie):
        l = len(word)
        if l > 1:
            self.get_pair_in_trie(word[0], word[1], trie)
            if self.check_tp(self.n_prefix, self.n_suffix) and self.get_bf(trie[word[0]]) <= self.bf: # verify if the
                self.suffixes.append(word[1])
                self.suffixes_bf.append(word[0] + str(self.get_bf(trie[word[0]])))
            else:
                self.suffixes[len(self.suffixes) - 1] = self.suffixes[len(self.suffixes) - 1] + word[1]
        if l > 2:
            if word[0] in trie.keys():
                self.split_suffix(word[1:], trie[word[0]])

    def get_pair_in_trie(self, prefix, suffix, trie):
        self.n_prefix = 0
        self.n_suffix = 0
        if prefix in trie:
            if suffix in trie[prefix]:
                self.n_prefix = trie[prefix]["##"]
                self.n_suffix = trie[prefix][suffix]["##"]

    def check_tp(self, m, d): # verify if Tolerance Principle applies
        if not m > 1:
            return False
        else:
            tp = m / log(m)
        if self.cutoff <= m != d > tp:
            return True
        else:
            return False

    def get_bf(self, m): # return the branching factor of the mother node
        keys = m.keys()
        n_keys = len(keys)
        for k in keys:
            if k in self.special_tokens:
                n_keys -= 1
        return n_keys

    def add_items_to_trie(self, items):
        for item in items:
            self.add_item_to_trie(item)

    def add_item_to_trie(self, item):
        if item not in self.last_item_in_trie:
            self.last_item_in_trie[item] = {}
        self.last_item_in_trie = self.last_item_in_trie[item]

    def pad_sentence(sentence, l):
        """
        Pads the given sentence with "[pad]" tokens at the beginning to reach the desired length.

        Parameters:
        - sentence (str): The original sentence to be padded.
        - l (int): The desired total number of tokens in the sentence after padding.

        Returns:
        - str: The padded sentence.
        """
        words = sentence.split()
        n_pad = max(l - len(words), 0)  # Ensure n_pad is not negative
        pad_tokens = ["[pad]"] * n_pad
        padded_sentence = ' '.join(pad_tokens + words)
        return padded_sentence

    def get_num_chars_in_trie(self):
        return self.num_chars_in_trie

    def get_num_chars_in_corpus(self):
        return self.num_chars_in_corpus

    def get_vocab_size(self):
        return self.idx

    def get_num_tokens_in_corpus(self):
        return self.num_tokens_in_corpus

    def get_num_types_in_corpus(self):
        return len(self.types)

    def get_compression_ratio(self):
        return round(self.num_chars_in_trie / self.num_chars_in_corpus, 3)

    def get_ttr(self):
        return round(len(self.types) / self.num_tokens_in_corpus, 3)

    def save_config(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + '/tokenizer.json', 'w') as f:
            json.dump(self.roots, f)

    def load_config(self, save_dir):
        with open(save_dir + '/tokenizer.json', 'r') as f:
            self.roots = json.load(f)

    def save_types(self, file):
        with open(file, 'w') as f:
            json.dump(self.types, f)


def sort_trie_by_freq(d):
    if not isinstance(d, dict):
        return d
    # Sort the dictionary items by the value of the nested key '##'
    sorted_items = sorted(
        d.items(),
        key=lambda item: item[1].get('##', float('-inf')) if isinstance(item[1], dict) else float('-inf'),
        reverse=True
    )
    # Clear the dictionary and update with sorted items
    d.clear()
    for k, v in sorted_items:
        d[k] = sort_trie_by_freq(v)
    return d


def find_idx_path(d, target_value, path=None):
    if path is None:
        path = []
    for key, value in d.items():
        if key == 'IDX' and value == target_value:
            return path
        elif isinstance(value, dict):
            result = find_idx_path(value, target_value, path + [key])
            if result is not None:
                return result
    return None
