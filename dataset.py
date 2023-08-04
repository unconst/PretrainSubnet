import os
import random
import requests
from tqdm import tqdm
from datasets import load_dataset
from urllib.parse import urlparse
from torch.utils.data import DataLoader

def get_next_dataloader( 
        tokenizer,
        batch_size: int = 1,
        sequence_length: int = 1024,
        mock: bool = False,
    ):

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation = True, padding = "max_length", max_length = sequence_length, return_tensors = "pt")
    if not mock:

         # Get all URLS.
        url = 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
        response = requests.get(url)
        all_urls = response.content.decode('utf-8').split('\n')

        # Create download folder.
        file = random.choice(all_urls)
        dload_loc = "~/" + urlparse(file).path[1:]
        os.makedirs(os.path.dirname(dload_loc), exist_ok=True) 

        # Download data.
        r = requests.get(file, stream=True)
        with open(dload_loc, 'wb') as f:
            for chunk in tqdm(r.iter_content(1024)):
                f.write(chunk)
            
        # Load dataset.
        dataset = load_dataset('json', data_files = dload_loc )

        # Shuffle dataset
        # shuffled_dataset = dataset.shuffle(buffer_size = 1000, seed = random.randint(0, 1000))

        # Tokenize the dataset.
        tokenized_dataset = dataset.map( tokenize_function, batched=True )

        # Create dataloader
        dataloader = DataLoader( tokenized_dataset, batch_size = batch_size)
    else:
        # Load the mock loader.
        texts = ["mock sentence "+str(i) for i in range(100)]  # creating 100 mock sentences
        encoded_texts = [tokenize_function({"text": txt}) for txt in texts]
        dataloader = DataLoader(encoded_texts, batch_size = batch_size)

    return dataloader
