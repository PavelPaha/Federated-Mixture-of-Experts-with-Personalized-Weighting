import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import random
from transformers import AutoTokenizer
from tqdm import tqdm

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


class WikiText103LMDataset(Dataset):
    def __init__(self, source, seq_len=40, shuffle=False):
        """
        source: HuggingFace dataset или любой iterable с ключом "text"
        seq_len: длина последовательности
        shuffle: тасовать данные при инициализации
        """
        self.source = source
        self.seq_len = seq_len
        self.shuffle = shuffle
        
        # Подготавливаем все последовательности заранее
        self.sequences = self._prepare_sequences()
        
        if self.shuffle:
            random.shuffle(self.sequences)

    def _prepare_sequences(self):
        """Подготавливает все последовательности заранее"""
        sequences = []
        
        # Собираем все тексты
        texts = []
        for example in tqdm(self.source, desc="Preparing sequences"):
            text = example["text"]
            if text and not text.isspace():
                texts.append(text)
        
        # Создаем поток токенов
        token_stream = []
        for text in tqdm(texts, desc="Tokenizing"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            token_stream.extend(ids)
        
        # Генерируем все последовательности
        i = 0
        while i + self.seq_len + 1 <= len(token_stream):
            chunk = token_stream[i:i+self.seq_len+1]
            inp = torch.tensor(chunk[:-1], dtype=torch.long)
            tgt = torch.tensor(chunk[1:], dtype=torch.long)
            sequences.append((inp, tgt))
            i += 1
            
        return sequences

    def __getitem__(self, idx):
        """Возвращает последовательность по индексу"""
        return self.sequences[idx]

    def __len__(self):
        """Возвращает точное количество последовательностей"""
        return len(self.sequences)


class StreamingWikiTextDataset(IterableDataset):
    """Полностью потоковый датасет без загрузки в память"""
    
    def __init__(self, source, seq_len=40, shuffle=False):
        self.source = source
        self.seq_len = seq_len
        self.shuffle = shuffle
        
    def __iter__(self):
        # Создаем поток токенов
        token_buffer = []
        
        for example in self.source:
            text = example["text"]
            if not text or text.isspace():
                continue
                
            # Токенизируем текст
            ids = tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(ids)
            
            # Генерируем последовательности пока есть достаточно токенов
            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[:self.seq_len + 1]
                token_buffer = token_buffer[1:]  # Сдвигаем на 1 токен
                
                inp = torch.tensor(chunk[:-1], dtype=torch.long)
                tgt = torch.tensor(chunk[1:], dtype=torch.long)
                yield inp, tgt
        
        # Обрабатываем оставшиеся токены
        while len(token_buffer) >= self.seq_len + 1:
            chunk = token_buffer[:self.seq_len + 1]
            token_buffer = token_buffer[1:]
            
            inp = torch.tensor(chunk[:-1], dtype=torch.long)
            tgt = torch.tensor(chunk[1:], dtype=torch.long)
            yield inp, tgt


def create_wikitext_dataloader(source, batch_size=6, seq_len=40, num_workers=1, shuffle=True, streaming=True):
    """
    Создает DataLoader для WikiText датасета
    
    Args:
        source: исходный датасет
        batch_size: размер батча
        seq_len: длина последовательности
        num_workers: количество воркеров
        shuffle: перемешивать ли данные
        streaming: использовать ли потоковый режим (экономит память)
    """
    if streaming:
        ds = StreamingWikiTextDataset(source, seq_len=seq_len, shuffle=shuffle)
        # Для IterableDataset shuffle должен быть False
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )
    else:
        ds = WikiText103LMDataset(source, seq_len=seq_len, shuffle=shuffle)
        # Для обычного Dataset можно использовать shuffle
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            shuffle=shuffle,
            pin_memory=True if torch.cuda.is_available() else False
        )
