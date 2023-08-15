import os
import json
import typing
import torch
import shutil
import random
import requests
import tempfile
import bittensor as bt
import zstandard as zstd
from tqdm import tqdm
from unittest.mock import patch
from datasets import load_dataset
from urllib.parse import urlparse
from transformers import AutoTokenizer

BASE_CACHE_DIR = "~/.cache/huggingface/datasets"
DATA_CACHE_DIR = None

def get_next_dataloader(
    tokenizer="gpt2",
    batch_size=1,
    sequence_length=1024,
    mock=False,
    shuffle_seed=42,
    load_script_path="scripts/red_pajama_loader.py",
    cache_dir="~/.cache/huggingface/datasets",
):
    """
    Load and tokenize datasets for training.

    Args:
    - tokenizer (str or tokenizer instance, default="gpt2"): Tokenizer identifier or actual tokenizer.
    - batch_size (int, default=1): Batch size for data loading.
    - sequence_length (int, default=1024): Maximum sequence length for tokenized data.
    - mock (bool, default=False): If True, mock data will be generated instead of loading actual data.
    - shuffle_seed (int, default=42): Seed value for shuffling the dataset.
    - load_script_path (str, default="scripts/red_pajama_loader.py"): Path to the data loading script.
    - cache_dir (str, default="~/.cache/huggingface/datasets"): Path to the cache directory.

    Returns:
    - tuple: (url, path, tokenized_data_generator).
    """

    global DATA_CACHE_DIR

    # Delete the old cache directory if it exists
    if DATA_CACHE_DIR and os.path.exists(DATA_CACHE_DIR): shutil.rmtree(DATA_CACHE_DIR)    

    # Create a new sub tempdir in the cache directory for this batch of data
    cache_dir = tempfile.mkdtemp( prefix='redpajama_', dir=os.path.expanduser( BASE_CACHE_DIR ) )
    DATA_CACHE_DIR = cache_dir

    # Load the dataset: Use the actual dataset if not in mock mode, otherwise generate mock data
    if not mock:
        dataset = load_dataset(load_script_path, name='default', cache_dir=cache_dir, split='train')
        dataset = dataset.shuffle(shuffle_seed)
    else:
        # Generate a list of mock sentences with variable repetitions
        dataset = [
            {"text": "mock sentence " + str(i) * random.randint(0, 1000)}
            for i in range(random.randint(1, 1000))
        ]

    # Initialize the tokenizer if a string identifier is provided
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    tokenized_data_generator = _tokenize_data(
        dataset,
        tokenizer=tokenizer,
        max_seq_length=sequence_length,
        batch_size=batch_size,
    )

    # Return information about this particular data subset
    url, _ = next(reversed(dataset._info.download_checksums.items()))
    size = dataset.download_size

    return size, url, tokenized_data_generator
    
def _tokenize_data(data, tokenizer, max_seq_length=512, batch_size=32):
    # Buffer to temporarily hold the tokenized data until a full batch is ready
    buffer = []
    
    # Iterate over each item in the dataset
    for text in data['text']:
        # Tokenize the text using the given tokenizer
        tokenized_text_full = tokenizer.encode_plus(text)
        
        # Extract input IDs and attention masks
        input_ids = tokenized_text_full['input_ids']
        attention = tokenized_text_full['attention_mask']

        # Lists to store split tokenized text and attention masks
        ids_list = []
        attention_list = []

        # Split the tokenized text and attention masks into segments of length `max_seq_length`
        for i in range(0, len(input_ids), max_seq_length):
            end = min(i + max_seq_length, len(input_ids))
            ids = input_ids[i:end]
            attention_mask = attention[i:end]

            # If a segment is shorter than `max_seq_length`, pad it
            if len(ids) < max_seq_length:
                ids += [tokenizer.pad_token_id] * (max_seq_length - len(ids))
                attention_mask += [0] * (max_seq_length - len(attention_mask))
            ids_list.append(ids)
            attention_list.append(attention_mask)

        # Add tokenized segments to the buffer
        for ids, attention_mask in zip(ids_list, attention_list):
            buffer.append({'input_ids': ids, 'attention_mask': attention_mask})

            # When the buffer reaches the batch size, convert to tensors and yield as a batch
            if len(buffer) == batch_size:
                input_ids_tensor = torch.tensor([item['input_ids'] for item in buffer])
                attention_mask_tensor = torch.tensor([item['attention_mask'] for item in buffer])
                yield {'input_ids': input_ids_tensor, 'attention_mask': attention_mask_tensor}
                buffer = [] # Clear buffer for the next batch

    # Handle any remaining items in the buffer
    if buffer:
        # Pad the remaining items to match the batch size
        padding_count = batch_size - len(buffer)
        pad_item = {'input_ids': [tokenizer.pad_token_id] * max_seq_length, 'attention_mask': [0] * max_seq_length}
        buffer.extend([pad_item] * padding_count)

        # Convert the buffer to tensors and yield as a batch
        input_ids_tensor = torch.tensor([item['input_ids'] for item in buffer])
        attention_mask_tensor = torch.tensor([item['attention_mask'] for item in buffer])
        yield {'input_ids': input_ids_tensor, 'attention_mask': attention_mask_tensor}



import unittest
from transformers import AutoTokenizer

class TestGetNextDataloader(unittest.TestCase):

    def test_mock_dataloader(self):
        # Test with the mock option enabled
        dataloader = get_next_dataloader(mock=True, batch_size=3, sequence_length=20)
        for batch in dataloader:
            self.assertEqual(batch["input_ids"].shape[0], 3)  # Checking batch size
            self.assertEqual(batch["input_ids"].shape[1], 20)  # Checking sequence length

    def test_tokenization_with_mock_data(self):
        # Test the tokenization process with mock data
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        dataloader = get_next_dataloader(mock=True, tokenizer=tokenizer, split="train", batch_size=2, sequence_length=20)
        for batch in dataloader:
            self.assertEqual(batch["input_ids"].shape[0], 2)  # Checking batch size
            self.assertEqual(batch["input_ids"].shape[1], 20)  # Checking sequence length

    def test_padding_last_batch_with_mock_data(self):
        # Test that the last batch is properly padded with mock data
        dataloader = get_next_dataloader(mock=True, split="train", batch_size=5, sequence_length=10)
        last_batch = None
        for batch in dataloader:
            last_batch = batch

        # Assuming the last batch may not be full, checking padding
        self.assertEqual(last_batch["input_ids"].shape[0], 5)  # Checking batch size
        self.assertEqual(last_batch["input_ids"].shape[1], 10)  # Checking sequence length

# If you want to run the tests:
if __name__ == '__main__':
    unittest.main()
