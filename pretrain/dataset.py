# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer

class Dataset:

    def __init__( self, config ):

        self.config = config
        
        # Define batch size.
        self.batch_size = self.config.bs

        # Load the 'wikitext' dataset
        self.wikitext_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')['train']

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Add a padding token and set it to the same as the EOS token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize the training_dataset
        def encode(examples):
            # Encode the text and generate a unique id for each example in the batch.
            encodings = self.tokenizer(examples['text'])
            return {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']}

        # Tokenize the training_dataset
        self.train_dataset = self.wikitext_dataset.map(encode, batched=True, remove_columns=['text'])

        # Add unique ids to the dataset
        self.train_dataset = self.train_dataset.map(lambda example, idx: {'id': idx}, with_indices=True)

        # Build the dataset collator.
        self.data_collator = DataCollatorForLanguageModeling( tokenizer = self.tokenizer, mlm = False, pad_to_multiple_of = 128 )

        # Create a DataLoader
        self.dataloader = DataLoader( self.train_dataset, batch_size = self.batch_size, shuffle = True, collate_fn = self.data_collator )
    