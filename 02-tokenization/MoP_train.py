import sys
import json
import MorPiece as MoP


def main():
    # Hyperparameters
    vocab_size = 100000
    cutoff = 8
    bf = 10
    min_frequency = 2

    # File paths
    corpus_file = sys.argv[1]  # e.g.: ../data/ita/processed/all_eng.txt
    output_dir = sys.argv[2]  # e.g.: ./tokenizer/

    with open(corpus_file, 'r', encoding='utf8') as f:
        text = f.read()

    # Tokenizer training
    mp = MoP.MorPiece(vocab_size=vocab_size, cutoff=cutoff, min_frequency=min_frequency, bf=bf)

    # mp.train(text)
    # mp.save_config(output_dir)
    # mp.save_types(output_dir + '/vocab_types.json')

    # Uncomment the following line and comment the previous three lines to load a pre-trained tokenizer
    mp.load_config(output_dir)

    # Tokenizer training information
    print(f"Number of tokens processed {mp.get_num_tokens_in_corpus()}")
    print(f"Number of types processed {mp.get_num_types_in_corpus()}")
    print(f"Number of chars in the root trie {mp.get_num_chars_in_trie()}")
    print(f"Vocabulary size {mp.get_vocab_size()}")
    print(f"Model and tokenizer saved in {output_dir}")

    for l in text.split("\n"):
        tokenized_data = mp.encode(l)
        contains_string = any(isinstance(item, str) for item in tokenized_data)
        if contains_string:
            print(l)
            print(tokenized_data)

    # to test the tokenizer
    # s = "the worker must go in the office where the dogs barks all days - long much longer than any impossible cats"
    # s = "io mangio tu mangi egli mangia noi mangiamo voi mangiate loro mangiano io lavo tu lavi lui lava noi laviamo voi lavate loro lavano lavarono laveranno laverebbero lavarlo"
    s = """interest in the chemistry of unhexquadium is largely prompted by predictions that the isotope 482uhq (
with 164 protons and 318 neutrons ) , would be at the center of a possible second island of stability ( the
first being centered on 306ubb or 298fl ) ."""
    # s = "laveranno"
    print("Sentence to tokenize: " + s)
    idx = mp.encode(s)
    print(mp.tokenized_words)
    print(idx)
    print(mp.decode(idx))


if __name__ == "__main__":
    main()