import os
import torch
import random
import requests
from tqdm import tqdm
from datasets import load_dataset
from urllib.parse import urlparse
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer


def get_next_dataloader(
    tokenizer="gpt2",
    load_script_path="load_redpajama_random.py",
    split='train',
    batch_size=1,
    sequence_length=1024,
    mock=False,
    return_dataset=False,
):
    if mock:
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation = True, padding = "max_length", max_length = sequence_length, return_tensors = "pt")
        # Load the mock loader.
        texts = ["mock sentence "+str(i) for i in range(100)]  # creating 100 mock sentences
        encoded_texts = [tokenize_function({"text": txt}) for txt in texts]
        dataloader = DataLoader(encoded_texts, batch_size = batch_size)
        return dataloader

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
                    yield BatchEncoding({'input_ids': input_ids_tensor, 'attention_mask': attention_mask_tensor})
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
            yield BatchEncoding({'input_ids': input_ids_tensor, 'attention_mask': attention_mask_tensor})

    dataset = load_dataset(load_script_path, 'default', split=split)

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_data_generator = _tokenize_data(
        dataset, 
        tokenizer=tokenizer, 
        max_seq_length=sequence_length
    )

    if return_dataset:
        return tokenized_data_generator, dataset

    return tokenized_data_generator