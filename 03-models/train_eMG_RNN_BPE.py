import os
import sys

import torch
import torch.nn as nn
from data_Batching import PrepareDataset
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_eMG_RNN import EMGLanguageModel
from tokenizer_BPE import BPE

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def calculate_accuracy(predictions, targets):
	_, predicted = torch.max(predictions, dim=1)
	correct = (predicted == targets).float()
	accuracy = correct.sum() / len(correct)
	return accuracy.item()


def collate_fn(batch):
	# Separate inputs and targets
	inputs, targets = zip(*batch)

	# Pad sequences
	inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
	targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100)  # Use -100 as padding for targets

	return inputs_padded, targets_padded


def evaluate(model, dataloader, criterion, device):
	model.eval()
	total_loss = 0
	total_accuracy = 0

	with torch.no_grad():
		for inputs, targets in dataloader:
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			if isinstance(outputs, tuple):
				output = outputs[0]
			else:
				output = outputs
			loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
			accuracy = calculate_accuracy(output.view(-1, output.size(-1)), targets.view(-1))
			total_loss += loss.item()
			total_accuracy += accuracy

	avg_loss = total_loss / len(dataloader)
	avg_accuracy = total_accuracy / len(dataloader)
	return avg_loss, avg_accuracy


# Training function
def train(model, dataloader, optimizer, criterion, device, scheduler=None, clip_grad_norm=1.0):
	model.train()
	total_loss = 0
	total_accuracy = 0
	iterations = len(dataloader)

	progress_bar = tqdm(dataloader, desc="Training", miniters=int(iterations / 100), mininterval=1800)

	for inputs, targets in progress_bar:
		inputs, targets = inputs.to(device), targets.to(device)

		optimizer.zero_grad()  # Zero gradients for each batch

		# Forward pass
		outputs = model(inputs)

		# Handle the case where model returns a tuple
		if isinstance(outputs, tuple):
			output = outputs[0]  # Assume the first element is the main output
		else:
			output = outputs

		mask = targets != -100
		loss = criterion(output[mask].view(-1, output.size(-1)), targets[mask].view(-1))

		# loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))

		loss.backward()  # Backward pass
		clip_grad_norm_(model.parameters(), clip_grad_norm)  # Gradient clipping
		optimizer.step()  # Optimizer step

		accuracy = calculate_accuracy(output[mask].view(-1, output.size(-1)), targets[mask].view(-1))  # Calculate accuracy

		total_loss += loss.item()
		total_accuracy += accuracy

		# Update progress bar
		progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'}, refresh=False)

	if scheduler:
		scheduler.step()

	avg_loss = total_loss / len(dataloader)
	avg_accuracy = total_accuracy / len(dataloader)

	return avg_loss, avg_accuracy


# Main function
def main():
	# Hyperparameters
	VOCAB_SIZE = 100000
	EMBEDDING_DIM = 650
	HIDDEN_DIM = 650
	NUM_LAYERS = 1
	BATCH_SIZE = 64
	NUM_EPOCHS = 100
	LEARNING_RATE = 0.002
	DROP_OUT = 0.2
	SEQ_LENGTH = 74
	REGIMEN = 'naturalistic'  # naturalistic = [[sentence_1], [sentence_2], ...];
	# conversational = [[sentence_1, sentence_2], [sentence_2, sentence_3], ...]
	# default/redundant = [[1, 2, ..., SEQ_LENGTH], [2, ..., SEQ_LENGTH+1], ...]

	# Main training loop with early stopping
	patience = 3
	best_val_loss = float('inf')
	epochs_without_improvement = 0

	cutoff = 100
	bf = 15
	min_frequency = 3
	add_special_tokens = False

	# arguments passed inline: fullpath/corpus_text_file, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, REGIMEN
	corpus_file = sys.argv[1]
	if sys.argv[2]:
		EMBEDDING_DIM = int(sys.argv[2])
	if sys.argv[3]:
		HIDDEN_DIM = int(sys.argv[3])
	if sys.argv[4]:
		NUM_LAYERS = int(sys.argv[4])
	if sys.argv[5]:
		REGIMEN = sys.argv[5]
	model_name = 'EN_100M_BPE_eMG_RNN_' + REGIMEN + '_E' + str(EMBEDDING_DIM) + '_H' + str(HIDDEN_DIM) + 'x' + str(NUM_LAYERS)

	# Train tokenizer
	tokenizer = BPE(min_frequency=min_frequency, vocab_size=VOCAB_SIZE)
	tokenizer.train(corpus_file)
	tokenizer.save_config(model_name)

	# Tokenizer training information
	print(f'Vocabulary size {tokenizer.get_vocab_size()}, min_frequency={min_frequency},  add_special_tokens={add_special_tokens}')
	print(f'Tokenizer config saved in {model_name}')

	# Create dataset and dataloader
	dataset = PrepareDataset(corpus_file, tokenizer, SEQ_LENGTH, add_special_tokens=add_special_tokens, regimen=REGIMEN)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8, collate_fn=collate_fn)

	# Initialize model
	model = EMGLanguageModel(tokenizer.get_vocab_size(), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROP_OUT)

	print(f'Building EMG network {model_name} with these hyperparameters:\nEMBEDDING_DIM={EMBEDDING_DIM}, HIDDEN_DIM={HIDDEN_DIM}, NUM_LAYERS={NUM_LAYERS}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, SEQ_LENGTH={SEQ_LENGTH}')

	# Multi-GPU setup
	if torch.cuda.device_count() > 1:
		print(f'Using {torch.cuda.device_count()} GPUs')
		model = nn.DataParallel(model)

	model = model.to(device)

	# Define loss and optimizer
	criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
	scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

	# Training loop
	for epoch in range(NUM_EPOCHS):
		loss, accuracy = train(model, dataloader, optimizer, criterion, device, scheduler)
		val_loss, val_accuracy = evaluate(model, dataloader, criterion, device)
		print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			epochs_without_improvement = 0

			# Optionally save the model checkpoint
			torch.save(model.state_dict(), os.path.join(model_name, str(epoch) + '_model.pth'))

			torch.save({'vocab_size': VOCAB_SIZE, 'embedding_dim': EMBEDDING_DIM, 'hidden_dim': HIDDEN_DIM, 'num_layers': NUM_LAYERS, 'batch_size': BATCH_SIZE, 'num_epochs': NUM_EPOCHS, 'learning_rate': LEARNING_RATE, 'seq_length': SEQ_LENGTH, 'state_dict': model.state_dict()},
				os.path.join(model_name, str(epoch) + '_model_with_params.pth'))

		else:
			epochs_without_improvement += 1  # Early stopping condition
		if epochs_without_improvement >= patience:
			print('Early stopping triggered. No improvement for', patience, 'epochs.')
			break

	# Save the model and tokenizer
	if not os.path.exists(model_name):
		os.makedirs(model_name)

	torch.save({'vocab_size': VOCAB_SIZE, 'embedding_dim': EMBEDDING_DIM, 'hidden_dim': HIDDEN_DIM, 'num_layers': NUM_LAYERS, 'batch_size': BATCH_SIZE, 'num_epochs': NUM_EPOCHS, 'learning_rate': LEARNING_RATE, 'seq_length': SEQ_LENGTH, 'state_dict': model.state_dict()},
		os.path.join(model_name, 'best_model_with_params.pt'))

	print(f'Model with parameters and tokenizer saved in {model_name}')


if __name__ == '__main__':
	main()