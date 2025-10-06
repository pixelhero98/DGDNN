"""Utility classes for generating rolling-window graph datasets.

The :class:`MyDataset` class wraps financial time-series data into
PyTorch-friendly graph samples that can be consumed by GNN models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.linalg import expm
from torch.utils.data import Dataset

__all__ = ["MyDataset"]


@dataclass(frozen=True)
class DatasetPaths:
    """Container tracking the relevant directories for a dataset build."""

    root: Path
    destination: Path

    @staticmethod
    def from_strings(root: str, destination: str) -> "DatasetPaths":
        """Utility helper to construct a :class:`DatasetPaths` object."""

        return DatasetPaths(Path(root).expanduser(), Path(destination).expanduser())


class MyDataset(Dataset):
    """Rolling-window graph dataset for financial time-series data.

    Each sample corresponds to ``window`` consecutive trading days for a
    collection of tickers.  The features are stacked ``Open``, ``High``,
    ``Low``, ``Close`` and ``Volume`` values which are log-transformed for
    numerical stability.  The adjacency matrix can either use a sparsified
    entropy-energy formulation or a heat-kernel approximation controlled by
    ``fast_approx``.
    """

    FEATURE_COLUMNS: Sequence[str] = ("Open", "High", "Low", "Close", "Volume")

    def __init__(
        self,
        root: str,
        dest: str,
        market: str,
        tickers: Sequence[str],
        start: str,
        end: str,
        window: int,
        mode: str = "train",
        fast_approx: bool = False,
        heat_tau: float = 5.0,
        sparsify_threshold: float = 0.3,
        log_eps: float = 1e-12,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if window < 2:
            msg = "Window length must be at least 2 to build features and labels."
            raise ValueError(msg)

        self.paths = DatasetPaths.from_strings(root, dest)
        self.market = market
        self.tickers = list(tickers)
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.window = window
        self.mode = mode
        self.fast_approx = fast_approx
        self.heat_tau = heat_tau
        self.sparsify_threshold = sparsify_threshold
        self.log_eps = log_eps
        self.norm_eps = norm_eps

        self._frames_full: Dict[str, pd.DataFrame] = {}
        self._frames_range: Dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            csv_path = self.paths.root / f"{market}_{ticker}_30Y.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"CSV file for ticker '{ticker}' was not found at {csv_path}"
                )
            frame = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
            self._frames_full[ticker] = frame
            self._frames_range[ticker] = frame.loc[self.start : self.end]

        if not self._frames_range:
            raise ValueError("No tickers were supplied to MyDataset.")

        self._dates = self._common_trading_days(self._frames_range.values())
        if len(self._dates) < self.window:
            raise ValueError("Not enough overlapping trading days to satisfy window size.")

        self._date_to_index = {date: idx for idx, date in enumerate(self._dates)}
        self._next_day = self._compute_next_day()
        self._features = self._stack_features()

        self.graph_dir = self._prepare_output_directory()
        self.graph_count = self._expected_graphs()

        if not self._graphs_exist():
            self._build_graphs()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return self.graph_count

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        graph_path = self.graph_dir / f"graph_{index}.pt"
        if not graph_path.exists():
            raise IndexError(f"Graph index {index} is out of bounds for {self.graph_count} samples.")
        return torch.load(graph_path)

    # ------------------------------------------------------------------
    # Helper construction routines
    # ------------------------------------------------------------------
    def _prepare_output_directory(self) -> Path:
        directory = (
            self.paths.destination
            / f"{self.market}_{self.mode}_{self.start.date()}_{self.end.date()}_{self.window}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _expected_graphs(self) -> int:
        return len(self._dates) - self.window + (1 if self._next_day is not None else 0)

    def _graphs_exist(self) -> bool:
        if self.graph_count <= 0:
            return False
        return all((self.graph_dir / f"graph_{i}.pt").exists() for i in range(self.graph_count))

    @staticmethod
    def _common_trading_days(frames: Iterable[pd.DataFrame]) -> List[pd.Timestamp]:
        common = None
        for frame in frames:
            dates = set(frame.index.normalize())
            common = dates if common is None else common & dates
        if not common:
            return []
        return sorted(common)

    def _compute_next_day(self) -> pd.Timestamp | None:
        after_sets = []
        for frame in self._frames_full.values():
            normalized = frame.index.normalize()
            after_sets.append(set(normalized[normalized > self.end]))
        if not after_sets:
            return None
        intersection = set.intersection(*after_sets)
        return min(intersection) if intersection else None

    def _stack_features(self) -> np.ndarray:
        stacked: List[np.ndarray] = []
        for date in self._dates:
            rows = [
                self._frames_range[ticker].loc[date, self.FEATURE_COLUMNS].to_numpy(dtype=float)
                for ticker in self.tickers
            ]
            stacked.append(np.stack(rows, axis=0))
        return np.stack(stacked, axis=0)

    # ------------------------------------------------------------------
    # Graph building
    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(values: np.ndarray) -> float:
        _, counts = np.unique(values, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def _adjacency(self, features: np.ndarray) -> torch.Tensor:
        node_count = features.shape[0]
        energy = np.einsum("ij,ij->i", features, features)
        entropy = np.apply_along_axis(self._entropy, 1, features)

        energy_ratio = energy[:, None] / (energy[None, :] + self.log_eps)
        entropy_sum = entropy[:, None] + entropy[None, :]

        tiled_i = np.repeat(features[:, None, :], node_count, axis=1)
        tiled_j = np.repeat(features[None, :, :], node_count, axis=0)
        joint_entropy = np.apply_along_axis(self._entropy, 2, np.concatenate((tiled_i, tiled_j), axis=-1))

        adjacency = energy_ratio * (np.exp(entropy_sum - joint_entropy) - 1.0)

        if self.fast_approx:
            augmented = adjacency + np.eye(node_count)
            degree_inv_sqrt = np.diag(1.0 / np.sqrt(augmented.sum(axis=1) + self.log_eps))
            heat_operator = degree_inv_sqrt @ augmented @ degree_inv_sqrt
            adjacency = expm(-self.heat_tau * (np.eye(node_count) - heat_operator))
        else:
            adjacency = np.where(adjacency >= self.sparsify_threshold, adjacency, 0.0)
            adjacency = np.log(adjacency + self.log_eps)

        adjacency = (adjacency + adjacency.T) / 2.0
        np.fill_diagonal(adjacency, 0.0)
        return torch.from_numpy(adjacency.astype(np.float32))

    def _build_graphs(self) -> None:
        close_idx = self.FEATURE_COLUMNS.index("Close")
        for graph_id in range(self.graph_count):
            date_slice = self._window_dates(graph_id)
            window_array = np.stack([self._fetch_date_block(date) for date in date_slice], axis=0)

            close_prices = window_array[:, :, close_idx]
            labels = (close_prices[-1] > close_prices[-2]).astype(np.int64)

            hist_window = window_array[:-1]
            node_count = hist_window.shape[1]
            node_features = np.log1p(hist_window.transpose(1, 0, 2).reshape(node_count, -1))

            adjacency = self._adjacency(node_features)
            graph = {
                "X": torch.from_numpy(node_features.astype(np.float32)),
                "A": adjacency,
                "Y": torch.from_numpy(labels),
            }

            torch.save(graph, self.graph_dir / f"graph_{graph_id}.pt")

    def _window_dates(self, graph_id: int) -> List[pd.Timestamp]:
        if graph_id < len(self._dates) - self.window:
            return self._dates[graph_id : graph_id + self.window + 1]
        if self._next_day is None:
            raise IndexError("Requested extra day for label but next day is unavailable.")
        return self._dates[-self.window :] + [self._next_day]

    def _fetch_date_block(self, date: pd.Timestamp) -> np.ndarray:
        if date in self._date_to_index:
            return self._features[self._date_to_index[date]]
        rows = []
        for ticker in self.tickers:
            row = self._frames_full[ticker].loc[date, self.FEATURE_COLUMNS].to_numpy(dtype=float)
            rows.append(row)
        return np.stack(rows, axis=0)

