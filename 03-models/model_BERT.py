import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tokenizers import BertWordPieceTokenizer
import os, sys
from transformers import AutoConfig, AutoModelForMaskedLM

class CustomBertForMaskedLM(BertForMaskedLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        state_dict = torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        return model

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.device_count()} GPU(s)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Train tokenizer
def train_tokenizer(corpus_file, vocab_size=100000, min_frequency=2):
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        strip_accents=False,
        lowercase=True,
    )

    tokenizer.train(
        files=[corpus_file],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        limit_alphabet=1000,
        wordpieces_prefix="##",
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAUSE]"],
    )

    return tokenizer

# Create dataset
class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        tokenized = self.tokenizer.encode(line, add_special_tokens=True, max_length=self.block_size, truncation=True)
        return torch.tensor(tokenized)

# Main training function
def train_bert(corpus_file, output_dir, num_train_epochs=3, per_device_train_batch_size=32):
    # Train tokenizer
    tokenizer = train_tokenizer(corpus_file)
    
    # Save tokenizer
    tokenizer.save_model(output_dir)
    tokenizer = BertTokenizerFast.from_pretrained(output_dir)

    # Prepare dataset
    dataset = TextDataset(tokenizer, corpus_file)
    
    # Configure BERT
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=512,
        type_vocab_size=1,
        is_decoder=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        model_type="bert",  # Explicitly set the model type
        architectures=["BertForMaskedLM"]  # Explicitly set the model architecture
    )
    
    model = BertForMaskedLM(config)

    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,  # Enable mixed-precision training
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    config.save_pretrained(output_dir)
    tokenizer.save_vocabulary(output_dir)
    
    # Create a custom_model.py file
    with open(os.path.join(output_dir, "custom_model.py"), "w") as f:
        f.write("""
from transformers import BertForMaskedLM, AutoConfig
import torch
import os

class CustomBertForMaskedLM(BertForMaskedLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        state_dict = torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        return model
""")

    # Create a README.md file with model information
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("# Custom BERT Masked Language Model\n\n")
        f.write("This is a custom BERT model trained for masked language modeling.\n")
        f.write("Use this model with the `CustomBertForMaskedLM` class from the `custom_model.py` file.\n")


if __name__ == "__main__":
    corpus_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_bert(corpus_file, output_dir)