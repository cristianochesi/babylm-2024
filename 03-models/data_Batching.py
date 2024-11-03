import torch
from torch.utils.data import Dataset

# Dataset class
class PrepareDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length, regimen='redundant',
                 add_special_tokens=False, start_token='[sos]', end_token='[eos]'):
        # Read the file and split into lines
        self.start_token = start_token
        self.end_token = end_token
        self.seq_length = seq_length
        self.regimen = regimen
        self.add_special_tokens=add_special_tokens
        self.tokenizer = tokenizer

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Clean and store lines using seq_length as max length to segment sentences
        self.lines = [' '.join(words[i:i+seq_length]) for line in lines if line.strip() for words in [line.strip().split()] for i in range(0, len(words), seq_length)]
        # Tokenize sequences
        self.tokenized_lines = [self.tokenizer.encode(line) for line in lines]
        self.all_tokenized_lines = [item for sublist in self.tokenized_lines for item in sublist]
        # print(self.all_tokenized_lines)

    def __len__(self):
        if self.regimen=='naturalistic' or self.regimen=='conversational':
            return len(self.tokenized_lines)
        else:
            return len(self.all_tokenized_lines)-self.seq_length

    def __getitem__(self, idx):
        if self.regimen=='naturalistic':
            input_sequence = self.tokenized_lines[idx]
        elif self.regimen=='conversational':
            line_1 = self.tokenized_lines[idx]
            if idx+1 < len(self.tokenized_lines):
                line_2 = self.tokenized_lines[idx+1]
            else: 
                line_2 = []
            input_sequence = line_1 + line_2
        else:
            input_sequence = self.all_tokenized_lines[idx:idx + self.seq_length+1]
        
        if self.add_special_tokens:
            input_sequence = self.tokenizer.encode(self.start_token) + input_sequence + self.tokenizer.encode(self.end_token)

        # Convert to tensors
        input_tensor = torch.tensor(input_sequence[:-1], dtype=torch.long)
        target_tensor = torch.tensor(input_sequence[1:], dtype=torch.long)
        
        # Pad sequences if they're shorter than seq_length
        if len(input_tensor) < self.seq_length:
            padding = torch.zeros(self.seq_length - len(input_tensor), dtype=torch.long)
            input_tensor = torch.cat([padding, input_tensor])
            target_tensor = torch.cat([padding, target_tensor])

        # print('INPUT:', input_tensor, self.tokenizer.decode(input_tensor))
        # print('TARGET:', target_tensor, self.tokenizer.decode(target_tensor))
        # print('\n')

        return input_tensor, target_tensor

