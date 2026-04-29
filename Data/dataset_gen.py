"""Utilities for constructing rolling-window financial graph datasets."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.linalg import expm
from sklearn.feature_selection import mutual_info_regression
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["MyDataset", "DatasetConfig"]


VALID_DATASET_MODES = {"train", "validation", "test"}
VALID_MI_METHODS = {"continuous", "discrete"}
VALID_FEATURE_TRANSFORMS = {"log", "ratio"}
GRAPH_MANIFEST_NAME = "graph_manifest.json"
GRAPH_MANIFEST_VERSION = 1


def _load_graph_payload(path: str) -> Dict[str, Tensor]:
    """Load a locally generated graph cache with the safest available torch API."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older PyTorch versions do not support weights_only.
        return torch.load(path, map_location="cpu")


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration describing how to construct :class:`MyDataset`."""

    root: str
    dest: str
    market: str
    tickers: List[str]
    start: str
    end: str
    window: int
    mode: str = "train"
    fast_approx: bool = False
    heat_tau: float = 5.0
    sparsify_threshold: float = 0.3
    log_eps: float = 1e-12
    norm_eps: float = 1e-6
    mi_method: str = "continuous"
    mi_neighbors: Optional[int] = None
    feature_transform: str = "ratio"
    positive_adj_filter: bool = False

    def __post_init__(self) -> None:
        if not self.root:
            raise ValueError("root directory must be provided")
        if not self.dest:
            raise ValueError("destination directory must be provided")
        if not self.tickers:
            raise ValueError("tickers list may not be empty")
        if self.window <= 1:
            raise ValueError("window must be greater than one to compute labels")

        try:
            start_ts = pd.to_datetime(self.start).normalize()
            end_ts = pd.to_datetime(self.end).normalize()
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError("start and end must be parseable as dates") from exc

        if start_ts > end_ts:
            raise ValueError("start date must be earlier than or equal to end date")

        mode = self.mode.lower()
        if mode not in VALID_DATASET_MODES:
            allowed = ", ".join(sorted(VALID_DATASET_MODES))
            raise ValueError(f"mode must be one of: {allowed}")

        mi_method = self.mi_method.lower()
        if mi_method not in VALID_MI_METHODS:
            allowed = ", ".join(sorted(VALID_MI_METHODS))
            raise ValueError(f"mi_method must be one of: {allowed}")

        feature_transform = self.feature_transform.lower()
        if feature_transform not in VALID_FEATURE_TRANSFORMS:
            allowed = ", ".join(sorted(VALID_FEATURE_TRANSFORMS))
            raise ValueError(f"feature_transform must be one of: {allowed}")

        if self.mi_neighbors is not None and self.mi_neighbors < 1:
            raise ValueError("mi_neighbors must be at least 1 when provided")

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "mi_method", mi_method)
        object.__setattr__(self, "feature_transform", feature_transform)


class MyDataset(Dataset[Dict[str, Tensor]]):
    """PyTorch dataset producing serialized rolling-window financial graphs.

    Each sample corresponds to a contiguous window of :math:`T` trading days for
    a set of companies.  The companies are represented as nodes in a graph whose
    adjacency is derived from the statistical similarity of their windowed
    feature trajectories.  The target label indicates whether the closing price
    of each company increased on the following day.
    """

    feature_columns: List[str] = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, config: Optional[DatasetConfig] = None, **kwargs: Any) -> None:
        super().__init__()

        if config is None:
            config = DatasetConfig(**kwargs)

        self.root = config.root
        self.dest = config.dest
        self.market = config.market
        self.tickers = list(config.tickers)
        self.start = pd.to_datetime(config.start).normalize()
        self.end = pd.to_datetime(config.end).normalize()
        self.window = int(config.window)
        self.mode = config.mode
        self.fast_approx = config.fast_approx
        self.heat_tau = float(config.heat_tau)
        self.sparsify_threshold = float(config.sparsify_threshold)
        self.log_eps = float(config.log_eps)
        self.norm_eps = float(config.norm_eps)
        self.mi_method = config.mi_method
        self.mi_neighbors = config.mi_neighbors
        self.feature_transform = config.feature_transform
        self.positive_adj_filter = bool(config.positive_adj_filter)

        if self.window <= 1:
            raise ValueError("window must be greater than one to compute labels")
        if not self.tickers:
            raise ValueError("tickers list may not be empty")

        self.data_frames_full: Dict[str, pd.DataFrame] = {}
        self.data_frames: Dict[str, pd.DataFrame] = {}

        for ticker in self.tickers:
            path = os.path.join(self.root, f"{self.market}_{ticker}_30Y.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"CSV file for ticker {ticker} not found at {path}"
                )

            df = pd.read_csv(path, parse_dates=[0], index_col=0)
            df.index = pd.to_datetime(df.index).normalize()
            df.sort_index(inplace=True)

            self.data_frames_full[ticker] = df
            self.data_frames[ticker] = df.loc[self.start : self.end]

        common_dates = self._compute_common_dates()
        if len(common_dates) < self.window:
            raise ValueError(
                "Insufficient overlapping data to construct the requested window"
            )

        self.full_dates = sorted(self._compute_full_common_dates())
        self.full_date_to_index = {
            date: index for index, date in enumerate(self.full_dates)
        }
        self.dates = sorted(common_dates)
        self.date_to_index = {date: idx for idx, date in enumerate(self.dates)}
        self.next_day = self._compute_next_day()

        self.features = self._stack_features(self.dates)

        self.output_directory = os.path.join(
            self.dest,
            f"{self.market}_{self.mode}_{self.start.date()}_{self.end.date()}_{self.window}",
        )
        self.manifest_path = os.path.join(self.output_directory, GRAPH_MANIFEST_NAME)
        os.makedirs(self.output_directory, exist_ok=True)

        self._ensure_serialized_graphs()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        total = len(self.dates) - self.window
        if self.next_day is not None:
            total += 1
        return max(total, 0)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:  # type: ignore[override]
        path = os.path.join(self.output_directory, f"graph_{index}.pt")
        if not os.path.exists(path):
            raise IndexError(f"Serialized sample {index} is missing at {path}")
        return _load_graph_payload(path)

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------
    def _compute_common_dates(self) -> set[pd.Timestamp]:
        common_dates: Optional[set[pd.Timestamp]] = None
        for frame in self.data_frames.values():
            dates = set(frame.index)
            common_dates = dates if common_dates is None else common_dates & dates
        return common_dates or set()

    def _compute_full_common_dates(self) -> set[pd.Timestamp]:
        common_dates: Optional[set[pd.Timestamp]] = None
        for frame in self.data_frames_full.values():
            dates = set(frame.index)
            common_dates = dates if common_dates is None else common_dates & dates
        return common_dates or set()

    def _compute_next_day(self) -> Optional[pd.Timestamp]:
        candidates: Optional[set[pd.Timestamp]] = None
        for frame in self.data_frames_full.values():
            future_dates = set(frame.index[frame.index > self.end])
            candidates = future_dates if candidates is None else candidates & future_dates
        return min(candidates) if candidates else None

    def _stack_features(self, dates: Iterable[pd.Timestamp]) -> np.ndarray:
        dates_index = pd.DatetimeIndex(list(dates))
        aligned_arrays = []
        for ticker in self.tickers:
            aligned = (
                self.data_frames[ticker]
                .reindex(dates_index)
                [self.feature_columns]
                .to_numpy(dtype=np.float64)
            )
            aligned_arrays.append(aligned)
        return np.stack(aligned_arrays, axis=1)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(arr: np.ndarray) -> float:
        values, counts = np.unique(arr, return_counts=True)
        probabilities = counts / counts.sum()
        return float(-np.sum(probabilities * np.log(probabilities + 1e-12)))

    def _mi_neighbor_count(self, sample_count: int) -> int:
        if sample_count <= 1:
            raise ValueError("continuous MI requires at least two feature samples")

        neighbors = (
            int(round(math.sqrt(sample_count)))
            if self.mi_neighbors is None
            else int(self.mi_neighbors)
        )
        return min(max(neighbors, 1), sample_count - 1)

    def _discrete_mutual_information(self, features: np.ndarray) -> np.ndarray:
        num_nodes = features.shape[0]
        entropy = np.apply_along_axis(self._entropy, 1, features)
        entropy_sum = entropy[:, None] + entropy[None, :]

        tiled_left = np.repeat(features[:, None, :], num_nodes, axis=1)
        tiled_right = np.repeat(features[None, :, :], num_nodes, axis=0)
        joint = np.concatenate([tiled_left, tiled_right], axis=-1)
        joint_entropy = np.apply_along_axis(self._entropy, 2, joint)

        return entropy_sum - joint_entropy

    def _continuous_mutual_information(self, features: np.ndarray) -> np.ndarray:
        num_nodes, sample_count = features.shape
        neighbors = self._mi_neighbor_count(sample_count)
        mutual_information = np.zeros((num_nodes, num_nodes), dtype=np.float64)

        for left in range(num_nodes):
            left_values = features[left].reshape(-1, 1)
            for right in range(left + 1, num_nodes):
                right_values = features[right].reshape(-1, 1)
                left_to_right = mutual_info_regression(
                    left_values,
                    features[right],
                    n_neighbors=neighbors,
                    random_state=0,
                )[0]
                right_to_left = mutual_info_regression(
                    right_values,
                    features[left],
                    n_neighbors=neighbors,
                    random_state=0,
                )[0]
                value = 0.5 * (left_to_right + right_to_left)
                mutual_information[left, right] = value
                mutual_information[right, left] = value

        return mutual_information

    def _mutual_information(self, features: np.ndarray) -> np.ndarray:
        if self.mi_method == "continuous":
            return self._continuous_mutual_information(features)
        if self.mi_method == "discrete":
            return self._discrete_mutual_information(features)
        raise ValueError(f"Unsupported mi_method: {self.mi_method}")

    def _feature_sample_count(self) -> int:
        return self.window * len(self.feature_columns)

    def _graph_fingerprint(self) -> Dict[str, Any]:
        sample_count = self._feature_sample_count()
        resolved_neighbors = (
            self._mi_neighbor_count(sample_count)
            if self.mi_method == "continuous"
            else None
        )
        return {
            "fingerprint_version": GRAPH_MANIFEST_VERSION,
            "mi_method": self.mi_method,
            "mi_neighbors": self.mi_neighbors,
            "resolved_mi_neighbors": resolved_neighbors,
            "feature_transform": self.feature_transform,
            "positive_adj_filter": self.positive_adj_filter,
            "feature_sample_count": sample_count,
            "sparsify_threshold": self.sparsify_threshold,
            "log_eps": self.log_eps,
            "norm_eps": self.norm_eps,
            "fast_approx": self.fast_approx,
            "heat_tau": self.heat_tau,
            "feature_columns": list(self.feature_columns),
            "node_count": len(self.tickers),
            "node_feature_shape": [len(self.tickers), sample_count],
        }

    def _adjacency(self, features: np.ndarray) -> Tensor:
        """Build an adjacency matrix from flattened node features."""

        num_nodes = features.shape[0]
        energy = np.einsum("ij,ij->i", features, features)
        energy_ratio = energy[:, None] / (energy[None, :] + self.log_eps)
        mutual_information = self._mutual_information(features)
        adjacency = energy_ratio * np.expm1(mutual_information)

        if self.fast_approx:
            adjacency_with_self = adjacency + np.eye(num_nodes)
            degree_inv = 1.0 / np.sqrt(adjacency_with_self.sum(axis=1) + self.log_eps)
            degree_inv_matrix = np.diag(degree_inv)
            heat_operator = degree_inv_matrix @ adjacency_with_self @ degree_inv_matrix
            adjacency = expm(-self.heat_tau * (np.eye(num_nodes) - heat_operator))
        else:
            adjacency[adjacency < self.sparsify_threshold] = 0.0
            adjacency = np.log(adjacency + self.log_eps)

        adjacency = (adjacency + adjacency.T) / 2.0
        np.fill_diagonal(adjacency, 0.0)
        if self.positive_adj_filter:
            off_diagonal = ~np.eye(num_nodes, dtype=bool)
            adjacency[(adjacency <= 0.0) & off_diagonal] = 0.0
            np.fill_diagonal(adjacency, 0.0)
        return torch.from_numpy(adjacency.astype(np.float32))

    def _ensure_serialized_graphs(self) -> None:
        total = len(self)
        if total == 0:
            return
        if not self._serialized_graphs_current(total):
            self._clear_serialized_graphs()
            self._build_graphs()
            self._write_graph_manifest(total)

    def _serialized_graphs_current(self, total: int) -> bool:
        manifest = self._read_graph_manifest()
        if manifest is None:
            return False
        if manifest.get("manifest_version") != GRAPH_MANIFEST_VERSION:
            return False
        if manifest.get("graph_count") != total:
            return False
        if manifest.get("fingerprint") != self._graph_fingerprint():
            return False
        return all(
            os.path.exists(os.path.join(self.output_directory, f"graph_{i}.pt"))
            for i in range(total)
        )

    def _read_graph_manifest(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.manifest_path):
            return None
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return manifest if isinstance(manifest, dict) else None

    def _write_graph_manifest(self, total: int) -> None:
        manifest = {
            "manifest_version": GRAPH_MANIFEST_VERSION,
            "graph_count": total,
            "fingerprint": self._graph_fingerprint(),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _clear_serialized_graphs(self) -> None:
        for filename in os.listdir(self.output_directory):
            if filename.startswith("graph_") and filename.endswith(".pt"):
                path = os.path.join(self.output_directory, filename)
                if os.path.isfile(path):
                    os.remove(path)
        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)

    def _build_graphs(self) -> None:
        total = len(self)
        for index in range(total):
            dates = self._select_dates(index)
            slice_array = self._collect_slice(dates)

            closes = slice_array[:, :, 3]
            labels = (closes[-1] > closes[-2]).astype(np.int64)

            window_array = slice_array[:-1]
            flattened = self._transform_features(window_array, dates[:-1])

            adjacency = self._adjacency(flattened)
            features = torch.from_numpy(flattened.astype(np.float32))

            payload = {
                "X": features,
                "A": adjacency,
                "Y": torch.from_numpy(labels),
            }
            torch.save(payload, os.path.join(self.output_directory, f"graph_{index}.pt"))

    def _transform_features(
        self, window_array: np.ndarray, window_dates: List[pd.Timestamp]
    ) -> np.ndarray:
        if self.feature_transform == "log":
            num_nodes = window_array.shape[1]
            return np.log1p(
                window_array.transpose(1, 0, 2).reshape(num_nodes, -1) + self.norm_eps
            )
        if self.feature_transform == "ratio":
            return self._ratio_features(window_array, window_dates)
        raise ValueError(f"Unsupported feature_transform: {self.feature_transform}")

    def _ratio_features(
        self, window_array: np.ndarray, window_dates: List[pd.Timestamp]
    ) -> np.ndarray:
        ratio_slices = []
        for offset, date in enumerate(window_dates):
            previous = self._previous_common_date(date)
            previous_values = self._collect_rows(previous)
            current_values = window_array[offset]
            denominator = np.maximum(np.abs(previous_values), self.norm_eps)
            ratio_slices.append((current_values - previous_values) / denominator)

        ratios = np.stack(ratio_slices, axis=0)
        num_nodes = ratios.shape[1]
        return ratios.transpose(1, 0, 2).reshape(num_nodes, -1)

    def _previous_common_date(self, date: pd.Timestamp) -> pd.Timestamp:
        if date not in self.full_date_to_index:
            raise ValueError(
                f"Ratio feature transform requires {date.date()} to be present "
                "for every ticker in the full CSV history"
            )
        index = self.full_date_to_index[date]
        if index == 0:
            raise ValueError(
                "Ratio feature transform requires a common previous trading date "
                f"before {date.date()}; include lookback rows before the dataset "
                "start date"
            )
        return self.full_dates[index - 1]

    # ------------------------------------------------------------------
    # Window helpers
    # ------------------------------------------------------------------
    def _select_dates(self, index: int) -> List[pd.Timestamp]:
        in_range = len(self.dates) - self.window
        if index < in_range:
            return self.dates[index : index + self.window + 1]
        if self.next_day is None:
            raise IndexError("Index exceeds available windows")
        return self.dates[-self.window :] + [self.next_day]

    def _collect_slice(self, dates: List[pd.Timestamp]) -> np.ndarray:
        slices = []
        for date in dates:
            if date in self.date_to_index:
                slices.append(self.features[self.date_to_index[date]])
                continue
            slices.append(self._collect_rows(date))
        return np.stack(slices, axis=0)

    def _collect_rows(self, date: pd.Timestamp) -> np.ndarray:
        rows = [
            self.data_frames_full[ticker]
            .loc[date, self.feature_columns]
            .to_numpy(dtype=np.float64)
            for ticker in self.tickers
        ]
        return np.stack(rows, axis=0)
