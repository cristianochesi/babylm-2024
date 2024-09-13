import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import os, sys
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")


# EMG baseline v.04.2
class EMGCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EMGCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, input, hidden):
        m_prev, c_prev = hidden

        # Combine input and previous hidden state
        merges = input + m_prev
        context = input + c_prev

        # Apply activations
        i = torch.tanh(merges)
        m = torch.sigmoid(merges)

        # Calculate output
        m_next = i * m
        c_next = torch.tanh(context) - i

        return m_next, c_next


class EMG(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EMG, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [EMGCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = [(torch.zeros(batch_size, self.hidden_size).to(x.device),
                       torch.zeros(batch_size, self.hidden_size).to(x.device)) for _ in range(self.num_layers)]

        mem_seq = []
        con_seq = []

        for t in range(seq_len):
            layer_input = x[:, t, :]
            mem_layers = []
            con_layers = []

            for layer_idx, cell in enumerate(self.cells):
                m_prev, c_prev = hidden[layer_idx]
                m_next, c_next = cell(layer_input, (m_prev, c_prev))
                hidden[layer_idx] = (m_next, c_next)
                layer_input = m_next

                mem_layers.append(m_next)
                con_layers.append(c_next)

            mem_seq.append(torch.stack(mem_layers, dim=1))
            con_seq.append(torch.stack(con_layers, dim=1))

        mem = torch.stack(mem_seq, dim=1)
        con = torch.stack(con_seq, dim=1)
        output = mem + torch.relu(con)

        return output, mem, con

# Define the LSTM model
class EMGLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(EMGLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.emg = EMG(embedding_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim * num_layers, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)

        output, mem, con = self.emg(embedded)

        # Reshape output to (batch_size, seq_len, num_layers * hidden_dim)
        output = output.view(batch_size, seq_len, -1)

        # Pass through the fully connected layer
        logits = self.fc(output)

        return logits, (mem[:, -1], con[:, -1])

# Dataset class
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

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
    _, predicted = torch.max(output, 1)
    correct = (predicted == targets).float().sum().item()
    total = targets.size(0)
    return correct / total


# Train tokenizer
def train_tokenizer(file_path, vocab_size=30000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"])

    tokenizer.train(files=[file_path], trainer=trainer)

    wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    wrapped_tokenizer.pad_token = "<pad>"
    wrapped_tokenizer.pad_token_id = tokenizer.token_to_id("<pad>")

    return wrapped_tokenizer


# Training function
def train(model, dataloader, optimizer, criterion, device, clip_grad_norm=1.0):
    model.train()
    total_loss = 0
    total_accuracy = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch, targets in progress_bar:
        batch, targets = batch.to(device), targets.to(device)

        # Zero gradients for each batch
        optimizer.zero_grad()

        # Forward pass
        output, _ = model(batch)
        loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        # Optimizer step
        optimizer.step()

        # Calculate accuracy
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
    EMBEDDING_DIM = 650
    HIDDEN_DIM = 650
    NUM_LAYERS = 4
    BATCH_SIZE = 64
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.001
    SEQ_LENGTH = 64

    # File paths
    corpus_file = sys.argv[1]
    if sys.argv[2]:
        EMBEDDING_DIM = int(sys.argv[2])
    if sys.argv[3]:
        HIDDEN_DIM = int(sys.argv[3])
    if sys.argv[4]:
        NUM_LAYERS = int(sys.argv[4])
    model_name = "EMG_04.2_E" + str(EMBEDDING_DIM) + "_H" +str(HIDDEN_DIM)+ "x" + str(NUM_LAYERS)

    # Train tokenizer
    tokenizer = train_tokenizer(corpus_file, VOCAB_SIZE)

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
    model = EMGLanguageModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
    model.padding_idx = tokenizer.pad_token_id

    print(
        f"Building EMG baseline network (EMG_04.02_BPE) with these hyperparameters:\nVOCAB_SIZE={VOCAB_SIZE}, EMBEDDING_DIM={EMBEDDING_DIM}, HIDDEN_DIM={HIDDEN_DIM}, NUM_LAYERS={NUM_LAYERS}")

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
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    torch.save(model.state_dict(), os.path.join(model_name, "model.pt"))
    tokenizer.save_pretrained(model_name)

    print(f"Model and tokenizer saved in {model_name}")


if __name__ == "__main__":
    main()