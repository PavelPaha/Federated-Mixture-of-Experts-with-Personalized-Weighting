# data.py
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import torch
import random
from typing import Tuple

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

class WikiText103LMDataset(Dataset):
    def __init__(self, source, seq_len=40, shuffle=False, stride=None):
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.stride = seq_len if stride is None else stride

        buffer = []
        for example in tqdm(source, desc="Tokenizing source"):
            text = example.get("text", None)
            if not text or text.isspace():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 0:
                continue
            buffer.extend(ids)

        self.chunks = []
        i = 0
        n = len(buffer)
        while i + seq_len + 1 <= n:
            self.chunks.append(buffer[i:i + seq_len + 1])
            i += self.stride

        if len(self.chunks) == 0:
            raise ValueError("No chunks created — check seq_len and dataset size")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        if self.shuffle:
            idx = random.randint(0, len(self.chunks) - 1)
        chunk = self.chunks[idx]
        inp = torch.tensor(chunk[:-1], dtype=torch.long)
        tgt = torch.tensor(chunk[1:], dtype=torch.long)
        return inp, tgt


def tokenize_and_group(dataset, seq_len, text_field="text"):
    def tokenize_function(examples):
        return tokenizer(examples[text_field], return_attention_mask=False)

    # Remove all original columns to avoid schema/length mismatches after grouping
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // seq_len) * seq_len
        result = {
            "input_ids": [concatenated[i:i + seq_len] for i in range(0, total_length, seq_len)]
        }
        return result

    # Overwrite tokenizer outputs with fixed-length chunks only
    grouped = tokenized.map(
        group_texts,
        batched=True,
        remove_columns=[c for c in tokenized.column_names if c != "input_ids"],
    )
    grouped.set_format(type="torch", columns=["input_ids"])
    return grouped

def lm_collate_from_grouped(batch):
    # Самый простой и быстрый collate_fn
    input_ids = torch.stack([item["input_ids"] for item in batch])
    return input_ids[:, :-1], input_ids[:, 1:]

def create_wikitext_dataloader_from_source(source, batch_size=6, seq_len=40, num_workers=1, shuffle=True, stride=None):
    """
    Use WikiText103LMDataset (concatenate tokens) — good for medium corpora.
    source: HF dataset split (iterable)
    """
    ds = WikiText103LMDataset(source, seq_len=seq_len, shuffle=shuffle, stride=stride)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle
    )

def create_dataloader_from_grouped(grouped_dataset, batch_size=8, shuffle=True, num_workers=0):
    return DataLoader(
        grouped_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lm_collate_from_grouped,
        pin_memory=True,
        drop_last=True
    )

def get_dataloaders(
    name: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    c4_subset_pct: float = 0.01
) -> Tuple[DataLoader, DataLoader, GPT2TokenizerFast]:
    name = name.lower()
    if name in ("wikitext-103", "wikitext103"):
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        train_split = dataset["train"]
        val_split = dataset["validation"]
        train_loader = create_wikitext_dataloader_from_source(train_split, batch_size=batch_size, seq_len=seq_len, num_workers=num_workers, shuffle=shuffle, stride=seq_len)
        val_loader = create_wikitext_dataloader_from_source(val_split, batch_size=batch_size, seq_len=seq_len, num_workers=num_workers, shuffle=False, stride=seq_len)
        return train_loader, val_loader, tokenizer

    if name in ("wikitext-2", "wikitext2", "wikitext-2-v1", "wikitext-2-raw-v1"):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_split = dataset["train"]
        val_split = dataset["validation"]
        # for small wikitext2 we can also use grouped map
        grouped_train = tokenize_and_group(train_split, seq_len)
        grouped_val = tokenize_and_group(val_split, seq_len)
        train_loader = create_dataloader_from_grouped(grouped_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = create_dataloader_from_grouped(grouped_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, tokenizer

    if name == "openwebtext":
        dataset = load_dataset("openwebtext")
        train_split = dataset["train"]
        # no validation split provided — we split train 99/1
        split = train_split.train_test_split(test_size=0.01, seed=42)
        train_split = split["train"]
        val_split = split["test"]
        grouped_train = tokenize_and_group(train_split, seq_len)
        grouped_val = tokenize_and_group(val_split, seq_len)
        train_loader = create_dataloader_from_grouped(grouped_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = create_dataloader_from_grouped(grouped_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, tokenizer

    if name == "c4":
        # take small percent to keep it light by default
        pct = max(0.0001, float(c4_subset_pct))
        # Use official C4 streaming/subset to avoid downloading full corpus
        dataset = load_dataset("c4", "en", split=f"train[:{pct*100}%]")
        # create grouped
        grouped = tokenize_and_group(dataset, seq_len)
        # further split grouped dataset into train/val
        split = grouped.train_test_split(test_size=0.01, seed=42)
        train_loader = create_dataloader_from_grouped(split["train"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = create_dataloader_from_grouped(split["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, tokenizer

    if name == "lm1b":
        # Try to load a ready-made parquet copy on the Hub (mirrors).
        # Many community uploads exist (parquet) because newer `datasets` versions
        # don't support old dataset script loaders anymore.
        hub_candidates = [
            "dvruette/lm1b",  # community mirror (parquet) — good first try
            "lm1b"             # legacy name (may fail if only script exists)
        ]
        dataset = None
        last_exc = None
        for cand in hub_candidates:
            try:
                dataset = load_dataset(cand)
                break
            except Exception as e:
                last_exc = e

        if dataset is None:
            # Final fallback: give actionable message (don't silently fail).
            raise RuntimeError(
                "Couldn't load LM1B via HuggingFace `datasets` API. "
                "Recent `datasets` releases disabled loading dataset scripts (lm1b.py). "
                "Options:\n"
                "  * use a hub mirror (e.g. 'dvruette/lm1b')\n"
                "  * install tensorflow_datasets and load via tfds (`tfds.load('lm1b')`),\n"
                "  * or downgrade `datasets` to <4.0.0 to keep old script support: "
                "`pip install \"datasets<4.0.0\"`.\n"
                f"Original error: {last_exc}"
            )

        # preserve previous logic: try 'validation' else 'test'
        train_split = dataset["train"]
        val_split = dataset["validation"] if "validation" in dataset else dataset["test"]

        grouped_train = tokenize_and_group(train_split, seq_len)
        grouped_val = tokenize_and_group(val_split, seq_len)
        train_loader = create_dataloader_from_grouped(grouped_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = create_dataloader_from_grouped(grouped_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, tokenizer

    raise ValueError(f"Unknown or unsupported dataset name: {name}")
