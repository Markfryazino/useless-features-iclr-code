import numpy as np
import torch
import os
import json
import logging
import inspect

from tqdm import tqdm, trange
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


PAD_ID = 0
BOS_ID = 1

# features[i] = features AFTER tokens_str[i] has been played
# legal_tokens[i] = legal tokens BEFORE tokens_str[i] has been played

@dataclass 
class RawSequence:
    tokens_str:       List[str]        # list of tokens as str (no <bos>) (seq_len - 1)
    legal_tokens_str: List[List[str]]  # list of list of legal tokens as str for each position (seq_len - 1)
    features:         np.array         # 2d array of features for each position (seq_len - 1 x feature_dim)
                                       # 1d array of loss mask for each position (seq_len - 1)
    loss_mask:        Optional[np.array] = None
    features_type:    Optional[np.array] = None


@dataclass
class SequenceInstance:
    tokens_str:     List[str]                # list of tokens as str (starting with <bos>) (seq_len)
    token_ids:      np.array                 # 1d array of token ids (starting with 1) (seq_len)
    legal_tokens:   np.array                 # 2d array of legal tokens for each position (seq_len - 1 x vocab_size)
    features:       np.array                 # 2d array of features for each position (seq_len x feature_dim)
    mask:           np.array                 # 1d array of mask for each position (1 if not padding, 0 if padding) (seq_len)
    loss_mask:      np.array                 # 1d array of loss mask for each position (1 if not ignore, 0 if ignore) (seq_len)
    aux:            Dict[str, Any] = None    # auxiliary data
    features_type:  Optional[np.array] = None


class Sampler:
    def __init__(self, seed: int, task_vocab: List[str]):
        self.generator = np.random.default_rng(seed)
        self.seed = seed

        if task_vocab is not None:
            assert "<pad>" not in task_vocab, "<pad> is a reserved token"
            assert "<bos>" not in task_vocab, "<bos> is a reserved token"
            
            vocab = ["<pad>", "<bos>"] + task_vocab
            self.vocab_size = len(vocab)
            self._token_to_id = {token: i for i, token in enumerate(vocab)}
            self._id_to_token = {i: token for i, token in enumerate(vocab)}

    def id2token(self, id: int) -> str:
        return self._id_to_token[id]

    def token2id(self, token: str) -> int:
        return self._token_to_id[token]

    def get_max_len(self) -> int:
        raise NotImplementedError("Subclass must implement this method")

    def _generate_raw_sequence(self) -> RawSequence:
        raise NotImplementedError("Subclass must implement this method")
    
    def _get_feature_names(self) -> List[str]:
        raise NotImplementedError("Subclass must implement this method")
    
    def get_custom_metrics(self, logits: torch.Tensor, tokens: torch.Tensor, legal_tokens: torch.Tensor) -> Dict[str, float]:
        return {} # Subclass can implement this method if needed

    def get_feature_name(self, id: int) -> str:
        return self._get_feature_names()[id]
    
    def get_feature_id(self, feature_name: str) -> int:
        return self._get_feature_names().index(feature_name)

    def generate_example(self, split: str) -> SequenceInstance:
        sig = inspect.signature(self._generate_raw_sequence)
        if 'split' in sig.parameters:
            example = self._generate_raw_sequence(split)
        else:
            example = self._generate_raw_sequence()

        tokens_str = ["<bos>"] + example.tokens_str
        token_ids = np.array([self.token2id(token) for token in tokens_str])

        legal_tokens = np.zeros((len(tokens_str) - 1, self.vocab_size), dtype=np.int32)

        for i, legal_tokens_str in enumerate(example.legal_tokens_str):
            for token in legal_tokens_str:
                legal_tokens[i, self.token2id(token)] = 1

        features = np.concatenate([np.zeros((1, self.feature_dim)), example.features])

        # propagate optional features_type similarly to features
        features_type = None
        if getattr(example, 'features_type', None) is not None:
            features_type = np.concatenate([np.zeros((1, self.feature_dim)), example.features_type])

        mask = token_ids != PAD_ID

        loss_mask = np.concatenate([np.ones(1), example.loss_mask]) \
                    if example.loss_mask is not None \
                    else np.ones(len(tokens_str))

        return SequenceInstance(
            tokens_str=tokens_str,
            token_ids=token_ids,
            legal_tokens=legal_tokens,
            features=features,
            mask=mask,
            loss_mask=loss_mask,
            features_type=features_type
        )
    
    def visualize_sequence(self, sequence: SequenceInstance):
        # Subclass can implement this method if needed
        return None
    

class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[SequenceInstance]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        ft = example.features_type if getattr(example, 'features_type', None) is not None else np.zeros_like(example.features)
        return {
            "token_ids": torch.LongTensor(example.token_ids),
            "legal_tokens": torch.LongTensor(example.legal_tokens),
            "features": torch.FloatTensor(example.features),
            "features_type": torch.tensor(ft, dtype=torch.uint8),
            "mask": torch.tensor(example.mask, dtype=torch.uint8),
            "loss_mask": torch.tensor(example.loss_mask, dtype=torch.int32)
        }
    
    def save(self, path: str):
        torch.save(self.examples, path)

    @classmethod
    def load(cls, path: str):
        examples = torch.load(path, weights_only=False)
        return cls(examples)


# A streaming dataset that writes examples to disk in fixed-size shards and
# loads them back lazily during iteration to keep memory usage low.
class ShardedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, shard_size: int, total_size: int, shard_dir: str, shard_prefix: str):
        self.shard_size = shard_size
        self.total_size = total_size
        self.shard_dir = shard_dir
        self.shard_prefix = shard_prefix
        os.makedirs(self.shard_dir, exist_ok=True)

        self._current_examples: List[SequenceInstance] = []
        # Discover any pre-existing shards for lazy loading
        self._shard_paths: List[str] = sorted([
            os.path.join(self.shard_dir, f)
            for f in os.listdir(self.shard_dir)
            if f.startswith(self.shard_prefix) and f.endswith(".pt")
        ])
        self._num_added = 0

    def _save_current_shard(self):
        if len(self._current_examples) == 0:
            return
        shard_idx = len(self._shard_paths)
        shard_path = os.path.join(self.shard_dir, f"{self.shard_prefix}_shard_{shard_idx:06d}.pt")
        torch.save(self._current_examples, shard_path)
        self._shard_paths.append(shard_path)
        self._current_examples = []

    def add_example(self, example: SequenceInstance):
        self._current_examples.append(example)
        self._num_added += 1
        if len(self._current_examples) >= self.shard_size:
            self._save_current_shard()

    def finalize(self):
        # Ensure any remaining examples are flushed to disk
        self._save_current_shard()

    def __iter__(self):
        # Iterate over saved shards first
        for shard_path in self._shard_paths:
            examples: List[SequenceInstance] = torch.load(shard_path, weights_only=False)
            for example in examples:
                ft = example.features_type if getattr(example, 'features_type', None) is not None else np.zeros_like(example.features)
                yield {
                    "token_ids": torch.LongTensor(example.token_ids),
                    "legal_tokens": torch.LongTensor(example.legal_tokens),
                    "features": torch.FloatTensor(example.features),
                    "features_type": torch.tensor(ft, dtype=torch.uint8),
                    "mask": torch.tensor(example.mask, dtype=torch.uint8),
                    "loss_mask": torch.tensor(example.loss_mask, dtype=torch.int32)
                }

        # Then yield any unsaved examples still in memory (if any)
        for example in self._current_examples:
            ft = example.features_type if getattr(example, 'features_type', None) is not None else np.zeros_like(example.features)
            yield {
                "token_ids": torch.LongTensor(example.token_ids),
                "legal_tokens": torch.LongTensor(example.legal_tokens),
                "features": torch.FloatTensor(example.features),
                "features_type": torch.tensor(ft, dtype=torch.uint8),
                "mask": torch.tensor(example.mask, dtype=torch.uint8),
                "loss_mask": torch.tensor(example.loss_mask, dtype=torch.int32)
            }

    def __len__(self):
        return self.total_size


# deduplication_size does not include <bos>!
class TrainEvalDataset:
    def __init__(
        self, sampler: Sampler, train_size: int, eval_size: int, deduplication_size: int = None,
        tqdm_on: bool = True, train_dataset: Optional[Dataset] = None, eval_dataset: Optional[Dataset] = None,
        sharding_size: Optional[int] = None, shards_dir: Optional[str] = None
    ):
        self.tqdm_on = tqdm_on
        self.sampler = sampler
        self.train_size = train_size
        self.eval_size = eval_size
        self.deduplication_size = deduplication_size
        self.sharding_size = sharding_size
        # When sharding is enabled, shards_dir must be provided (equal to dataset_path)
        self.shards_dir = shards_dir
        if self.sharding_size is not None and self.shards_dir is None:
            raise ValueError("Sharded dataset requires dataset_path (used as shards_dir). Please set cfg.dataset_path.")

        if train_dataset is not None and eval_dataset is not None:
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
        else:
            self.train_dataset, self.eval_dataset = self.generate_train_eval_dataset()

    def generate_train_eval_dataset(self):
        eval_examples = [self.sampler.generate_example(split="eval") for _ in trange(self.eval_size, disable=not self.tqdm_on, desc="eval dataset gen")]

        if self.deduplication_size is not None:
            prohibited_seqs = set([
                "".join([str(el) for el in example.tokens_str[:self.deduplication_size + 1]])
                for example in eval_examples
            ])

        eval_dataset = Dataset(eval_examples)

        # If sharding is off, return standard in-memory datasets
        if self.sharding_size is None:
            train_examples = []

            with tqdm(total=self.train_size, disable=not self.tqdm_on, desc="train dataset gen") as pbar:
                while len(train_examples) < self.train_size:
                    example = self.sampler.generate_example(split="train")

                    if self.deduplication_size is not None:
                        seq_str = "".join([str(el) for el in example.tokens_str])
                        if seq_str in prohibited_seqs:
                            continue

                    train_examples.append(example)
                    pbar.update(1)

            return Dataset(train_examples), eval_dataset

        # Sharding path: write both train and eval shards into the same directory with prefixes
        os.makedirs(self.shards_dir, exist_ok=True)

        train_dataset = ShardedIterableDataset(
            self.sharding_size,
            self.train_size,
            self.shards_dir,
            shard_prefix=f"train_{self.sampler.seed}"
        )
        with tqdm(total=self.train_size, disable=not self.tqdm_on, desc="train dataset gen (sharded)") as pbar:
            added = 0
            while added < self.train_size:
                example = self.sampler.generate_example(split="train")

                if self.deduplication_size is not None:
                    seq_str = "".join([str(el) for el in example.tokens_str])
                    if seq_str in prohibited_seqs:
                        continue

                train_dataset.add_example(example)
                added += 1
                pbar.update(1)
        train_dataset.finalize()

        return train_dataset, eval_dataset

    def save(self, path: str, identifier: str):
        os.makedirs(path, exist_ok=True)

        # If using in-memory datasets, persist as single .pt files for compatibility
        if isinstance(self.train_dataset, Dataset) and isinstance(self.eval_dataset, Dataset):
            self.train_dataset.save(os.path.join(path, "train.pt"))
            self.eval_dataset.save(os.path.join(path, "eval.pt"))
        else:
            # Sharded train dataset: ensure shards are flushed and metadata is saved
            if isinstance(self.train_dataset, ShardedIterableDataset):
                self.train_dataset.finalize()
            # Eval must be saved as a single file
            if isinstance(self.eval_dataset, Dataset):
                self.eval_dataset.save(os.path.join(path, "eval.pt"))

            meta = {
                "sharded": True,
                "train_size": self.train_size,
                "eval_size": self.eval_size,
                "shard_size": self.sharding_size,
            }
            with open(os.path.join(path, "meta.json"), "w") as mf:
                json.dump(meta, mf)

        with open(os.path.join(path, "identifier.txt"), "w") as f:
            f.write(identifier)

    @classmethod
    def load(cls, path: str, sampler: Sampler):
        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as mf:
                meta = json.load(mf)

            eval_dataset = Dataset.load(os.path.join(path, "eval.pt"))
            if meta.get("sharded", False):
                train_dataset = ShardedIterableDataset(
                    meta["shard_size"], meta["train_size"], path, shard_prefix=f"train"
                )
                return cls(
                    sampler=sampler,
                    train_size=meta["train_size"],
                    eval_size=meta["eval_size"],
                    deduplication_size=None,
                    tqdm_on=True,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    sharding_size=meta["shard_size"],
                    shards_dir=path
                )

        train_dataset = Dataset.load(os.path.join(path, "train.pt"))
        eval_dataset = Dataset.load(os.path.join(path, "eval.pt"))

        return cls(
            sampler=sampler,
            train_size=len(train_dataset),
            eval_size=len(eval_dataset),
            deduplication_size=None,
            tqdm_on=True,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

    @classmethod
    def load_or_save(cls, dir: str, sampler: Sampler, identifier: str,
                     train_size, eval_size, deduplication_size, tqdm_on,
                     sharding_size: Optional[int] = None):
        os.makedirs(dir, exist_ok=True)      

        identifier_json = eval(identifier)  

        matching_dirs = []
        
        for folder in os.listdir(dir):
            try:
                with open(os.path.join(dir, folder, "identifier.txt")) as f:
                    saved_identifier = f.read()
            except:
                saved_identifier = "{}"

            saved_identifier_json = eval(saved_identifier)
            
            if saved_identifier_json.keys() == identifier_json.keys() and all(str(saved_identifier_json[k]) == str(identifier_json[k]) for k in saved_identifier_json):
                matching_dirs.append(os.path.join(dir, folder))

        if len(matching_dirs) == 1:
            logging.info(f"Loading dataset from {matching_dirs[0]}")
            loaded = cls.load(matching_dirs[0], sampler)
            #assert (loaded.train_size == train_size) and (loaded.eval_size == eval_size), \
            #    f"Loaded dataset has different size than expected: {loaded.train_size} != {train_size} or {loaded.eval_size} != {eval_size}"
            return loaded

        elif len(matching_dirs) > 1:
            raise ValueError(f"Multiple matching datasets found for identifier {identifier}")
        
        logging.info(f"No matching dataset found for identifier {identifier}, generating and saving")
        # generating new folder number ahead of time so we can direct shard outputs there
        previous_numbers = [int(folder) for folder in os.listdir(dir) if folder.isdigit()]
        new_number = max(previous_numbers) + 1 if previous_numbers else 1
        save_root = os.path.join(dir, str(new_number))

        shards_dir = save_root if sharding_size is not None else None
        dataset = cls(
            sampler=sampler,
            train_size=train_size,
            eval_size=eval_size,
            deduplication_size=deduplication_size,
            tqdm_on=tqdm_on,
            sharding_size=sharding_size,
            shards_dir=shards_dir
        )
        dataset.save(save_root, identifier)

        return dataset
