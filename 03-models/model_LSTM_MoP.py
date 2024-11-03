import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tokenizer_MorPiece as MoP
import os, sys, re
from tqdm import tqdm

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the LSTM model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # print("Input shape:", x.shape)
        embedded = self.embedding(x)
        # print("Embedded shape:", embedded.shape)
        output, hidden = self.lstm(embedded, hidden)
        # print("LSTM output shape:", output.shape)
        output = self.fc(output)
        return output, hidden


# Dataset class
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        text = ''

        with open(file_path, 'r', encoding='utf-8') as f:
            # text=f.read()
            for line in f:
                text += '[sos] ' + re.sub('\n', ' [eos]\n', line)
        for item in tokenizer.encode(text):
            if isinstance(item, str):
                token = tokenizer.decode([item])
                print(f"Type = {type(item)}, Value = {item}, Token = {token}")
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_length - 1)

    def __getitem__(self, idx):
        if idx + self.seq_length + 1 > len(self.data):
            idx = len(self.data) - self.seq_length - 1

        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        return input_seq, target_seq


def calculate_accuracy(output, targets):
    predictions = output.argmax(dim=-1)
    return (predictions == targets).float().mean().item()


# Train tokenizer
def train_tokenizer(file_path, vocab_size=30000, min_frequency=2, cutoff=8):
    tokenizer = MoP.MorPiece(vocab_size=vocab_size, cutoff=cutoff, min_frequency=min_frequency)
    with open(file_path, 'r', encoding='utf8') as f:
        text = f.read()
    tokenizer.train(text)
    tokenizer.save_config('./tokenizer')
    print(f"Number of tokens processed {tokenizer.get_num_tokens_in_corpus()}")
    print(f"Number of types processed {tokenizer.get_num_types_in_corpus()}")
    print(f"Number of chars in the root trie {tokenizer.get_num_chars_in_trie()}")
    print(f"Vocabulary size {tokenizer.get_vocab_size()}")
    return tokenizer


# Training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_accuracy = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch, targets in progress_bar:
        batch, targets = batch.to(device), targets.to(device)
        optimizer.zero_grad()
        # print(f"Batch shape: {batch.shape}\n{batch.view}")
        # print(f"Targets shape: {targets.shape}\n{targets.view}")

        output, _ = model(batch)
        loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        # Print the output shape for debugging
        # print(f"Output shape: {output.shape}")

        accuracy = calculate_accuracy(output.view(-1, output.size(-1)), targets.view(-1))

        total_loss += loss.item()
        total_accuracy += accuracy

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    return avg_loss, avg_accuracy


# Main function
def main():
    # Hyperparameters
    VOCAB_SIZE = 100000
    CUTOFF = 100
    MIN_FREQUENCY = 2
    EMBEDDING_DIM = 650
    HIDDEN_DIM = 650
    NUM_LAYERS = 2
    BATCH_SIZE = 64
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.001
    SEQ_LENGTH = 64

    # File paths
    corpus_file = sys.argv[1]
    output_dir = sys.argv[2]

    # Train tokenizer
    tokenizer = train_tokenizer(corpus_file, vocab_size=VOCAB_SIZE, cutoff=CUTOFF, min_frequency=MIN_FREQUENCY)

    # Create dataset and dataloader
    dataset = TextDataset(corpus_file, tokenizer, SEQ_LENGTH)
    dataloader = DataLoader(dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=8, 
        drop_last=True
    )

    # Initialize model
    model = LSTMLanguageModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
    # model.padding_idx = tokenizer.

    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        loss, accuracy = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the model and tokenizer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    tokenizer.save_config(output_dir)

    print(f"Model and tokenizer saved in {output_dir}")

if __name__ == "__main__":
    main()