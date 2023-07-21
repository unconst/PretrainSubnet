# GPT2 Specific.
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer

def get_dataloader( self ) -> DataLoader:

    # Define batch size.
    batch_size = 16

    # Load the 'wikitext' dataset
    wikitext_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')['train']

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Add a padding token and set it to the same as the EOS token
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the training_dataset
    def encode(examples): return tokenizer(examples['text'])
    train_dataset = wikitext_dataset.map( encode, batched = True, remove_columns = ['text']) 

    # Build the dataset collator.
    data_collator = DataCollatorForLanguageModeling( tokenizer = tokenizer, mlm = False, pad_to_multiple_of = 128 )

    # Create a DataLoader
    dataloader = DataLoader( train_dataset, batch_size = batch_size, shuffle = True, collate_fn = data_collator )

    return dataloader, tokenizer
 