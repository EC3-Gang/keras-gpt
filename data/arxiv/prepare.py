import os
from tqdm import tqdm
import numpy as np
import tiktoken
import requests
from datasets import load_dataset

num_proc = 8

num_proc_load_dataset = num_proc

def download(file):
    urls_file_url = file

    with open(urls_file_url, 'r') as f:
        download_directory = 'downloaded_files'
        os.makedirs(download_directory, exist_ok=True)

        url_lines = [line.rstrip('\n') for line in f]

        for line in tqdm(url_lines, desc="Downloading", unit="file"):
            dload_loc = line.replace('https://data.together.xyz/redpajama-data-1T/v1.0.0/', '')
            dload_path = os.path.join(download_directory, os.path.dirname(dload_loc))
            os.makedirs(dload_path, exist_ok=True)
            
            file_url = line
            file_path = os.path.join(download_directory, dload_loc)
            
            response = requests.get(file_url)
            with open(file_path, 'wb') as f:
                f.write(response.content)

        print("Downloaded files successfully.")

        return download_directory

if __name__ == '__main__':
    os.environ["RED_PAJAMA_DATA_DIR"] = download('urls.txt')
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T", num_proc=num_proc_load_dataset)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') 
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['text']) 
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
