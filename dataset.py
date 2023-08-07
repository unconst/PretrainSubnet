import os
import json
import typing
import torch
import shutil
import random
import requests
from tqdm import tqdm
import zstandard as zstd
from unittest.mock import patch
from datasets import load_dataset
from urllib.parse import urlparse
from transformers import AutoTokenizer

DATA_URL = 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
DATA_DIR = "~/.cache/huggingface/datasets"

def get_next_dataloader(
    tokenizer = "gpt2",
    batch_size=1,
    sequence_length=1024,
    mock=False,
    shuffle_seed=42,
    return_dataset=False,
):
    index, path = load_new_files()
    if not mock:
        dataset = load_dataset('json', data_files = path)
        dataset = dataset.shuffle( shuffle_seed )
    else:
        dataset = [{"text": "mock sentence " + str(i) * random.randint( 0, 1000 ) } for i in range(random.randint( 1, 1000 ))]  # creating 100 mock sentences

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_data_generator = _tokenize_data(
        dataset, 
        tokenizer = tokenizer, 
        max_seq_length = sequence_length,
        batch_size = batch_size
    )
    if return_dataset:
        return tokenized_data_generator, dataset
    
    return index, path, tokenized_data_generator

def load_new_files() -> typing.Tuple[int, str]:
    try:
        shutil.rmtree(os.path.expanduser(DATA_DIR))
    except: pass
    os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True) 
    # Get all URLS.
    response = requests.get(DATA_URL)
    all_urls = response.content.decode('utf-8').split('\n')

    # Create download folder.
    index = random.choice(range(len(all_urls)))
    file = all_urls[index]
    encrypted_path = os.path.expanduser( DATA_DIR + '/' + urlparse(file).path[1:] )
    os.makedirs(os.path.dirname(encrypted_path), exist_ok=True) 

    print(encrypted_path)
    # Download data.
    r = requests.get(file, stream=True)
    with open(encrypted_path, 'wb') as f:
        for chunk in tqdm(r.iter_content(4096)):
            f.write(chunk)
    
    # Define the input and output file paths
    if encrypted_path.endswith('.zst'):
        decrypted_path = encrypted_path.rsplit('.', 1)[0]  # Removes the last extension (.zst)
        print(decrypted_path)

        # Open the input file for reading and the output file for writing
        # with zstd.open(open(encrypted_path, "rb"), "rt", encoding="utf-8") as encrypted_file, open(decrypted_path, 'w', encoding="utf-8") as decrypted_file:
        #     for i, row in tqdm(enumerate( encrypted_file )):
        #         data = json.loads(row)
        #         # Write the row to the output file
        #         decrypted_file.write(json.dumps(data) + '\n') 

        with open(encrypted_path, "rb") as encrypted_file, open(decrypted_path, 'wb') as decrypted_file:
            # Create a zstandard decompression stream
            dctx = zstd.ZstdDecompressor()
            # Use the copy_stream method for direct decompression
            dctx.copy_stream(encrypted_file, decrypted_file)
    else:
        decrypted_path = encrypted_path

    return index, decrypted_path
    
def _tokenize_data(data, tokenizer, max_seq_length=512, batch_size=32):
    # Buffer to temporarily hold the tokenized data until a full batch is ready
    buffer = []
    
    # Iterate over each item in the dataset
    for item in data:
        # Extract the text content from the item
        text = item["text"]
        
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
