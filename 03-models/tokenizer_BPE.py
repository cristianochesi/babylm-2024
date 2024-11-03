from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


class BPE:
	def __init__(self, vocab_size=100000, min_frequency=2):
		self.vocab_size = vocab_size
		self.min_frequency = min_frequency

	def train(self, file_path):
		tokenizer = Tokenizer(models.BPE())

		tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
		trainer = trainers.BpeTrainer(special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"], min_frequency=self.min_frequency, vocab_size=self.vocab_size)

		tokenizer.train(files=[file_path], trainer=trainer)

		wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
		wrapped_tokenizer.pad_token = "<pad>"
		wrapped_tokenizer.pad_token_id = tokenizer.token_to_id("<pad>")
		self.tokenizer = wrapped_tokenizer

		return self.tokenizer

	def save_config(self, model_name):
		self.tokenizer.save_pretrained(model_name)

	def encode(self, text):
		return self.tokenizer.encode(text.strip())

	def decode(self, idx):
		return self.tokenizer.decode(idx)

	def get_vocab_size(self):
		return len(self.tokenizer.get_vocab())