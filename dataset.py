import os
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

    def _tokenize_data(data, tokenizer, max_seq_length=512):
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            tokenized_batch = tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                pad_to_max_length=True,
                return_tensors="pt"
            )
            yield tokenized_batch

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