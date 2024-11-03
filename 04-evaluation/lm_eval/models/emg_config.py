import os

import torch
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class EMGCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.2):
        super(EMGCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define linear transformations
        self.input_transform = nn.Linear(input_size, hidden_size)
        self.hidden_transform = nn.Linear(hidden_size, 2 * hidden_size)

        # Define normalization and dropout layers
        # self.input_norm = nn.LayerNorm(hidden_size)
        # self.hidden_norm = nn.LayerNorm(2 * hidden_size)
        # self.dropout = nn.Dropout(p=dropout_rate) # dropout dramatically reduces the training performance!

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.zeros_(self.input_transform.bias)
        nn.init.xavier_uniform_(self.hidden_transform.weight)
        nn.init.zeros_(self.hidden_transform.bias)

    def forward(self, input, hidden):
        h_prev, c_prev = hidden

        # Transform and normalize input
        transformed_input = self.input_transform(input)
        # transformed_input = self.input_norm(transformed_input)
        # transformed_input = self.dropout(transformed_input)

        # Apply transformations and normalize
        gates = self.hidden_transform(h_prev)
        # gates = self.hidden_norm(gates)

        # Split gates and apply activations
        retain_gate, merge_gate = torch.chunk(gates, 2, dim=-1)
        retain_gate = torch.sigmoid(retain_gate)
        merge_gate = torch.sigmoid(merge_gate)

        r = retain_gate * transformed_input

        # Calculate next cell and hidden states
        c_next = torch.tanh(c_prev + r)
        m_next = (1 - merge_gate) * transformed_input +  merge_gate * c_next

        return m_next, c_next


class EMG(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(EMG, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([EMGCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = [(torch.zeros(batch_size, self.hidden_size, device=x.device), torch.zeros(batch_size, self.hidden_size, device=x.device)) for _ in range(self.num_layers)]

        outputs = []

        for t in range(seq_len):
            layer_input = x[:, t, :]
            layer_outputs = []

            for layer_idx, cell in enumerate(self.cells):
                m_prev, c_prev = hidden[layer_idx]
                m_next, c_next = cell(layer_input, (m_prev, c_prev))
                hidden[layer_idx] = (m_next, c_next)
                layer_input = m_next
                layer_outputs.append(m_next)

            outputs.append(torch.stack(layer_outputs, dim=1))

        output = torch.stack(outputs, dim=1)
        mem = output
        con = torch.stack([h[1] for h in hidden], dim=1).unsqueeze(1).expand(-1, seq_len, -1, -1)

        final_output = output

        return final_output, mem, con


class EMGLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=650, hidden_dim=650, num_layers=1, dropout_rate=0.2):
        super(EMGLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.emg = EMG(embedding_dim, hidden_dim, num_layers, dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, mem, con = self.emg(embedded, hidden)
        output = output.view(-1, self.hidden_dim)
        logits = self.fc(output)

        # Reshape logits back to (batch_size, seq_len, vocab_size)
        logits = logits.view(x.size(0), x.size(1), -1)

        return logits, (mem[:, -1], con[:, -1])