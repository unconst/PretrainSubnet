import os
import sys
import torch
import argparse
import bittensor as bt
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
from torch.nn import functional as F

def calculate_wikitext_perplexity( 
        model: torch.nn.Module, 
        tokenizer, 
        device: str,
        sequence_length: int = 1024,
    ):

    # Load the wiki test test.
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', 'test')
    text = ''.join(dataset['test']['text'])

    # Encode the text.
    encodings = tokenizer(text, return_tensors='pt')
    max_length = sequence_length
    stride = max_length // 2
    lls = []

    for i in tqdm( range(0, encodings.input_ids.size(1), stride) ):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))

        trg_len = end_loc - i
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids.to(device), labels=target_ids.to(device))
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    perplexity = torch.exp(torch.stack(lls).sum() / end_loc)
    return perplexity.item()
