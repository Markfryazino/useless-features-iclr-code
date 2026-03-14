from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import torch

from src.data_gen import Sampler, RawSequence, SequenceInstance


class OthelloState:
    """
    Represents the state of an Othello game.
    The board is from the perspective of the current player.
    +1 for the current player's stones, -1 for the opponent's.
    """
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),         (0, 1),
                  (1, -1),  (1, 0), (1, 1)]

    def __init__(self, board: np.ndarray):
        self.board = board

    def legal_actions(self) -> List[Tuple[int, int]]:
        """Returns a list of legal moves for the current player."""
        actions = []
        n = self.board.shape[0]
        for r in range(n):
            for c in range(n):
                if self.board[r, c] == 0 and self._can_capture(r, c):
                    actions.append((r, c))
        return actions

    def _can_capture(self, r: int, c: int) -> bool:
        """Checks if placing a stone at (r, c) captures any opponent stones."""
        return any(self._can_capture_in_direction(r, c, dr, dc) for dr, dc in self.directions)

    def _can_capture_in_direction(self, r: int, c: int, dr: int, dc: int) -> bool:
        """Checks for captures in a single direction."""
        i, j = r + dr, c + dc
        n = self.board.shape[0]
        found_opponent = False
        while 0 <= i < n and 0 <= j < n:
            if self.board[i, j] == -1:
                found_opponent = True
                i += dr
                j += dc
            elif self.board[i, j] == 1:
                return found_opponent
            else:
                break
        return False

    def move(self, action: Tuple[int, int]) -> 'OthelloState':
        """
        Applies a move, flips stones, and returns the new state for the next player.
        """
        r, c = action
        new_board = self.board.copy()
        new_board[r, c] = 1
        self._flip_stones(new_board, r, c)
        return OthelloState(-new_board)

    def _flip_stones(self, board: np.ndarray, r: int, c: int):
        """Flips opponent stones in all valid directions."""
        for dr, dc in self.directions:
            if self._can_capture_in_direction(r, c, dr, dc):
                self._flip_stones_in_direction(board, r, c, dr, dc)

    def _flip_stones_in_direction(self, board: np.ndarray, r: int, c: int, dr: int, dc: int):
        """Flips opponent stones in a single direction."""
        i, j = r + dr, c + dc
        stones_to_flip = []
        n = board.shape[0]
        while 0 <= i < n and 0 <= j < n:
            if board[i, j] == -1:
                stones_to_flip.append((i, j))
                i += dr
                j += dc
            elif board[i, j] == 1:
                for ri, cj in stones_to_flip:
                    board[ri, cj] = 1
                break
            else:
                break
    
    def get_features(self) -> np.ndarray:
        """Returns the flattened board as features."""
        return self.board.flatten()
    
    def get_features_type(self) -> np.ndarray:
        """
        Returns a flattened array (same shape as features) with values per cell:
        0 for empty cells; for occupied cells, 1 if inverting that cell does not
        change the set of legal actions for the current player, 2 otherwise.
        """
        n = self.board.shape[0]
        reference_actions = set(self.legal_actions())
        result = np.zeros_like(self.board, dtype=int)
        for r in range(n):
            for c in range(n):
                cell_value = self.board[r, c]
                if cell_value == 0:
                    result[r, c] = 0
                    continue
                new_board = self.board.copy()
                new_board[r, c] = -cell_value
                new_state = OthelloState(new_board)
                new_actions = set(new_state.legal_actions())
                result[r, c] = 1 if new_actions == reference_actions else 2
        return result.flatten()
    
    def clone(self) -> 'OthelloState':
        """Creates a deep copy of the current state."""
        return OthelloState(self.board.copy())


class OthelloSampler(Sampler):
    def __init__(self, seed: int, size: int, max_moves: int):
        self.size = size
        task_vocab = ["pass"] + [f"({r},{c})" for r in range(size) for c in range(size)]
        super().__init__(seed, task_vocab)
        self.feature_dim = size * size
        self.max_moves = max_moves

    def _get_feature_names(self) -> List[str]:
        return [f"({r},{c})" for r in range(self.size) for c in range(self.size)]

    def get_max_len(self) -> int:
        # Max moves is size*size - 4. Sequence includes <bos>.
        return min(self.max_moves, self.size * self.size - 4 + 1)

    def _initial_state(self) -> OthelloState:
        """Creates the standard initial Othello board."""
        board = np.zeros((self.size, self.size), dtype=int)
        center = self.size // 2
        board[center - 1, center - 1] = 1
        board[center, center] = 1
        board[center - 1, center] = -1
        board[center, center - 1] = -1
        return OthelloState(board)

    def _generate_raw_sequence(self, split: str) -> RawSequence:
        """Generates a random game of Othello."""
        state = self._initial_state()
        
        tokens_str = []
        legal_tokens_str = []
        features = []
        features_type = []  # if split == "eval" else None

        for _ in range(self.get_max_len()):
            legal_actions = state.legal_actions()

            current_legal_tokens = [f"({r},{c})" for r, c in legal_actions] if legal_actions else ["pass"]
            legal_tokens_str.append(current_legal_tokens)
            
            if not legal_actions:
                # If no moves, check if opponent can move. If so, pass.
                opponent_state = OthelloState(-state.board)
                action_str = "pass"
                state = opponent_state
            else:
                action_tuple = legal_actions[self.generator.choice(len(legal_actions))]
                action_str = f"({action_tuple[0]},{action_tuple[1]})"
                state = state.move(action_tuple)
            
            tokens_str.append(action_str)
            features.append(state.get_features())
            if features_type is not None:
                features_type.append(state.get_features_type())
        
        return RawSequence(
            tokens_str=tokens_str,
            legal_tokens_str=legal_tokens_str,
            features=np.array(features),
            features_type=np.array(features_type) if features_type is not None else None
        )

    def _plot_board(self, board: np.ndarray, step: int, plot_types: bool = False):
        """Plots the Othello board for a given state.
        If plot_types is True, overlays red crosses on cells with feature type 2.
        """
        original_board = board.copy()
        board = original_board * (1 - 2 * (step % 2))
        n = board.shape[0]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title(f'Othello Board at Step {step}')
        # Draw the grid
        for x in range(n + 1):
            ax.plot([x, x], [0, n], color='black')
        for y in range(n + 1):
            ax.plot([0, n], [y, y], color='black')
        # Place stones
        for x in range(n):
            for y in range(n):
                stone = board[x, y]
                if stone != 0:
                    circle = plt.Circle((y + 0.5, n - x - 0.5), 0.4,
                                        color='black' if stone == 1 else 'white',
                                        ec='black')
                    ax.add_artist(circle)
        # Overlay crosses for feature type 2 cells
        if plot_types:
            types = OthelloState(original_board).get_features_type().reshape(n, n)
            for x in range(n):
                for y in range(n):
                    if types[x, y] == 2:
                        ax.plot([y, y + 1], [n - x, n - x - 1], color='red', linewidth=1.5)
                        ax.plot([y, y + 1], [n - x - 1, n - x], color='red', linewidth=1.5)
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.gca().invert_yaxis()
        return fig

    def visualize_sequence(self, sequence: SequenceInstance, plot_types: bool = False):
        """Visualizes the game by plotting the board at each state."""
        figs = []
        initial_board = self._initial_state().board
        figs.append(self._plot_board(initial_board, step=0, plot_types=plot_types))
        
        for i, features in enumerate(sequence.features[1:]): # skip first feature which is zero
            board = features.reshape(self.size, self.size)
            figs.append(self._plot_board(board, step=i + 1, plot_types=plot_types))
        return figs

    def get_custom_metrics(self, logits: torch.Tensor, tokens: torch.Tensor, legal_tokens: torch.Tensor) -> Dict[str, float]:
        return {}
