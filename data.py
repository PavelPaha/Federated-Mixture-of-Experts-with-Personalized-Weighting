from torch.utils.data.dataset import IterableDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
import torch

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

class WikiText103LMIterable(IterableDataset):
    def __init__(self, source, seq_len=40):
        super().__init__()
        self.seq_len = seq_len
        self.raw_ds = source

    def __iter__(self):
        buffer = []
        for example in self.raw_ds:
            text = example["text"]
            if not text or text.isspace():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                inp = torch.tensor(chunk[:-1], dtype=torch.long)
                tgt = torch.tensor(chunk[1:], dtype=torch.long)
                yield inp, tgt


def create_wikitext_dataloader(source, batch_size=6, seq_len=40, num_workers=1):
    ds = WikiText103LMIterable(source, seq_len=seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=2,
    )