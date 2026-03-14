from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple, Dict
from datasets import DatasetDict
from transformers import AutoTokenizer

from src.data_gen import Sampler, RawSequence, SequenceInstance
from src.samplers.linguistics import tokens_str_to_features, features_to_vector, TOP_TAGS, TOP_DEPS


class LanguageSampler(Sampler):
    def __init__(self, seed: int, tokenizer_path: str, max_len: int, dataset_path: str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.vocab_size = len(tokenizer)

        super().__init__(seed, None)
        self.tokenizer = tokenizer

        self.max_len = max_len
        self.dataset_path = dataset_path

        self.dataset = DatasetDict.load_from_disk(dataset_path)

        self.iterators = {
            "train": iter(self.dataset["train"]),
            "eval": iter(self.dataset["validation"])
        }

    def _get_feature_names(self) -> List[str]:
        return [f"is_{tag}" for tag in TOP_TAGS] + [f"is_{dep}" for dep in TOP_DEPS] + ["pos_int", "pos_prop"]

    def get_max_len(self) -> int:
        return self.max_len

    def _generate_raw_sequence(self) -> RawSequence:
        raise NotImplementedError("Not implemented, use generate_example instead")
    
    def generate_example(self, split: str) -> SequenceInstance:
        example_length = 0
        while example_length < self.max_len:
            try:
                example = next(self.iterators[split])
                example_length = len(example["tokens"])
            except StopIteration:
                dataset_name = "train" if split == "train" else "validation"
                self.iterators[split] = iter(self.dataset[dataset_name])

        full_features = tokens_str_to_features(example["tokens_str"])

        starting_token_position = self.generator.integers(0, example_length - self.max_len + 1)
        tokens_str = example["tokens_str"][starting_token_position:starting_token_position + self.max_len]
        token_ids = example["tokens"][starting_token_position:starting_token_position + self.max_len]
        mask = np.ones(self.max_len, dtype=np.int32)
        loss_mask = np.ones(self.max_len, dtype=np.int32)
        partial_features = full_features[starting_token_position:starting_token_position + self.max_len]

        return SequenceInstance(
            tokens_str=tokens_str,
            token_ids=token_ids,
            mask=mask,
            loss_mask=loss_mask,
            legal_tokens=np.array([]),
            features=features_to_vector(partial_features),
            aux=partial_features
        )

    def get_custom_metrics(self, logits: torch.Tensor, tokens: torch.Tensor, legal_tokens: torch.Tensor) -> Dict[str, float]:
        return {}