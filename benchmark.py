import torch
import argparse
import bittensor as bt
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
from torch.nn import functional as F


# Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset', type=str, default = 'wikitext-2', choices=['wikitext-2'], help = 'Dataset to run benchmark on in [wikitext-2]')
    parser.add_argument( '--model', type=str, default = 'gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'mock'], help = 'Dataset to run benchmark on.')
    bt.logging.add_args( parser )
    return bt.config( parser )

config = parse_arguments()
bt.logging( config = config )
bt.logging.info( config )
pass

# Load model and tokenizer
def load_model_and_tokenizer():
    # Load pre-trained model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(config.model)
        model = GPT2LMHeadModel.from_pretrained(config.model) 
    elif config.model == 'gpt2-medium':
        tokenizer = GPT2Tokenizer.from_pretrained(config.model)
        model = GPT2LMHeadModel.from_pretrained(config.model)
    elif config.model == 'gpt2-large':
        tokenizer = GPT2Tokenizer.from_pretrained(config.model)
        model = GPT2LMHeadModel.from_pretrained(config.model)
    elif config.model == 'mock': 
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel(GPT2Config(n_layer = 1, n_head = 1))
    else:
        raise ValueError(f'Invalid model: {config.model}')
    model = model.to(device)
    model.eval()
    return model, tokenizer, device
model, tokenizer, device = load_model_and_tokenizer()
pass

# Load datasets
datasets = {
    'wikitext-2': load_dataset('wikitext', "wikitext-2-raw-v1", split="test"),
}

# Function to compute perplexity
def compute_perplexity(model, dataset):
    
    # Load encodings.
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    # Compute perplexity.
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).mean())

# Run benchmark
bt.logging.info(f'Running benchmark on {config.dataset}')
perplexity = compute_perplexity(model, datasets[config.dataset])
bt.logging.success(f'{config.dataset}: {perplexity.item()}')
