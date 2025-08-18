from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import torch
import random

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

class WikiText103LMDataset(Dataset):
    def __init__(self, source, seq_len=40, shuffle=False):
        """
        source: HuggingFace dataset или любой iterable с ключом "text"
        seq_len: длина последовательности
        shuffle: тасовать данные при каждом доступе
        """
        self.seq_len = seq_len
        self.shuffle = shuffle

        # собираем все токены в один массив
        buffer = []
        for example in tqdm(source):
            text = example["text"]
            if not text or text.isspace():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(ids)

        # разбиваем на куски seq_len+1
        self.chunks = []
        i = 0
        while i + seq_len + 1 <= len(buffer):
            self.chunks.append(buffer[i:i+seq_len+1])
            i += seq_len + 1

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        if self.shuffle:
            idx = random.randint(0, len(self.chunks)-1)
        chunk = self.chunks[idx]
        inp = torch.tensor(chunk[:-1], dtype=torch.long)
        tgt = torch.tensor(chunk[1:], dtype=torch.long)
        return inp, tgt


def create_wikitext_dataloader(source, batch_size=6, seq_len=40, num_workers=1, shuffle=True):
    ds = WikiText103LMDataset(source, seq_len=seq_len, shuffle=shuffle)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle
    )
