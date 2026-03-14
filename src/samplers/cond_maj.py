from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple, Dict

from src.data_gen import Sampler, RawSequence


class CondMajSampler(Sampler):
    def __init__(self, seed: int, vocab_size: int, input_size: int, output_size: int, exclude0: bool = False):
        self._vocab_size = vocab_size
        self.game_tokens = [f"<{i}>" for i in range(self._vocab_size)]
        super().__init__(seed, self.game_tokens)

        self.input_size = input_size
        self.output_size = output_size
        
        # self.feature_dim = 2 * self._vocab_size
        self.feature_dim = self._vocab_size
        self._game_token_to_idx = {token: i for i, token in enumerate(self.game_tokens)}
        self.key_token = self.game_tokens[0]
        self.exclude0 = exclude0

    def get_max_len(self) -> int:
        return self.input_size + self.output_size

    def _get_feature_names(self) -> List[str]:
        prev_is_features = [f"prev_is_{token}" for token in self.game_tokens]
        # cond_majority_features = [f"is_cond_majority_{token}" for token in self.game_tokens]
        return prev_is_features #+ cond_majority_features

    def _token2one_hot(self, token: str) -> np.array:
        features = np.zeros(self._vocab_size, dtype=np.int32)

        if token is not None:
            features[self._game_token_to_idx[token]] = 1
        return features

    def _counts2cond_majority_features(self, counts: np.array) -> np.array:
        max_count = np.max(counts)
        return (counts == max_count).astype(np.int32)

    def _counts2legal_tokens(self, counts: np.array) -> List[str]:
        max_count = np.max(counts) if not self.exclude0 else np.max(counts[1:])

        if not self.exclude0:
            majority_tokens = [token for token, count in zip(self.game_tokens, counts) if count == max_count]
        else:
            majority_tokens = [token for token, count in zip(self.game_tokens[1:], counts[1:]) if count == max_count]
        return majority_tokens
    
    def _step(self, new_token: str, tokens_str: List[str], features: np.array, counts: np.array) -> Tuple[List[str], np.array, np.array]:
        prev_token = None
        if len(tokens_str) > 0:
            prev_token = tokens_str[-1]

        if len(tokens_str) < self.input_size and prev_token == self.key_token:
            counts[self._game_token_to_idx[new_token]] += 1
        
        tokens_str.append(new_token)
        
        prev_is_features = self._token2one_hot(prev_token)
        # cond_majority_features = self._counts2cond_majority_features(counts)
        
        # combined_features = np.concatenate([prev_is_features, cond_majority_features])
        features[len(tokens_str) - 1, :] = prev_is_features
        # features[len(tokens_str) - 1, :] = combined_features
        
        return tokens_str, features, counts

    def _generate_raw_sequence(self) -> RawSequence:
        tokens_str = []
        legal_tokens_str = []
        features = np.zeros((self.input_size + self.output_size, self.feature_dim), dtype=np.int32)
        counts = np.zeros(self._vocab_size, dtype=np.int32)
        
        for _ in range(self.input_size):
            legal_tokens_str.append(self.game_tokens)
            new_token = self.generator.choice(self.game_tokens)
            tokens_str, features, counts = self._step(new_token, tokens_str, features, counts)

        for _ in range(self.output_size):
            majority_tokens = self._counts2legal_tokens(counts)
            legal_tokens_str.append(majority_tokens)
            new_token = self.generator.choice(majority_tokens)
            tokens_str, features, counts = self._step(new_token, tokens_str, features, counts)

        return RawSequence(
            tokens_str=tokens_str,
            legal_tokens_str=legal_tokens_str,
            features=features
        )
    
    def get_custom_metrics(self, logits: torch.Tensor, tokens: torch.Tensor, legal_tokens: torch.Tensor) -> Dict[str, float]:
        prediction = logits[:, :-1].argmax(dim=-1)
        legal_pred = torch.gather(legal_tokens, 2, prediction.unsqueeze(-1)).squeeze(-1)
        legal_output = legal_pred[:, -self.output_size:]
        legal_output_acc = (legal_output == 1).float().mean().item()
        first_output_token_acc = (legal_pred[:, -self.output_size] == 1).float().mean().item()

        return {
            "legal_output_acc": legal_output_acc,
            "first_output_token_acc": first_output_token_acc
        }
