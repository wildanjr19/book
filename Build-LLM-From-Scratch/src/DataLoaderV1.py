import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) # tokenizer the entire text

        # proses looping dengan sliding windows
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        # return total data
    def __len__(self):
        return len(self.input_ids)
        
        # return data
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_data_loader_v1(txt, batch_size = 4, max_length = 256, 
                          stride = 128, shuffle = True, drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding('gpt2') # initialize tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) # create dataset
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader