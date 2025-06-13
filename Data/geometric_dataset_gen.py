import os
import re
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.linalg import expm


class MyDataset(Dataset):
    """
    A PyTorch Dataset for rolling-window graph snapshots of company time series.

    Parameters:
      - root (str): directory containing CSVs named {market}_{ticker}_30Y.csv
      - dest (str): output directory for serialized graphs
      - market (str): market code prefix in filenames (e.g., 'NASDAQ')
      - start (str): inclusive window start date 'YYYY-MM-DD'
      - end (str): inclusive window end date 'YYYY-MM-DD'
      - window (int): number of past days T to use per graph
      - mode (str, optional): subfolder label, e.g. 'train' or 'test' (default 'train')
      - fast_approx (bool, optional): whether to use heat-kernel approximation (default False)

    Automatically discovers tickers by scanning CSV filenames in `root`.

    Graph dict entries:
      - X: Tensor [N, F * T] node features (log1p of F=5 features over T days per node)
      - A: Tensor [N, N] adjacency matrix (log-scaled or heat-kernel)
      - Y: Tensor [N] integer labels (count of days price rose in window)

    Example:
      >>> dataset = MyDataset(
      ...     root='data/csvs',
      ...     dest='data/graphs',
      ...     market='NASDAQ',
      ...     start='2020-01-01',
      ...     end='2020-12-31',
      ...     window=10,
      ... )
    """
    def __init__(
        self,
        root: str,
        dest: str,
        market: str,
        start: str,
        end: str,
        window: int,
        mode: str = 'train',
        fast_approx: bool = False,
    ):
        super().__init__()
        self.root, self.dest = root, dest
        self.market = market
        self.start, self.end = pd.to_datetime(start), pd.to_datetime(end)
        self.window = window
        self.mode = mode
        self.fast_approx = fast_approx

        # Discover tickers from filenames
        pattern = re.compile(rf"^{re.escape(market)}_(.+)_30Y\.csv$")
        self.tickers = []
        for fname in os.listdir(root):
            match = pattern.match(fname)
            if match:
                self.tickers.append(match.group(1))
        self.tickers.sort()
        N = len(self.tickers)

        # Load each ticker's DataFrame once
        self.data_frames = {}
        for t in self.tickers:
            path = os.path.join(root, f"{market}_{t}_30Y.csv")
            df = pd.read_csv(path, parse_dates=[0], index_col=0)
            df = df.loc[self.start:self.end]
            self.data_frames[t] = df

        # Common trading dates
        common = set.intersection(*[set(df.index.normalize()) for df in self.data_frames.values()])
        self.dates = sorted(common)

        # Next common date after end
        all_dates = set.intersection(*[
            set(df.index.normalize().strftime('%Y-%m-%d'))
            for df in self.data_frames.values()
        ])
        future = [d for d in all_dates if d > end]
        self.next_day = min(future) if future else None

        # Stack features: shape (n_dates, N, F)
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.features = np.stack([
            self.data_frames[t][feature_cols].values for t in self.tickers
        ], axis=1)

        # Prepare output directory
        out_dir = os.path.join(dest, f"{market}_{mode}_{self.start.date()}_{self.end.date()}_{window}")
        os.makedirs(out_dir, exist_ok=True)

        # Build graphs if missing
        if not all(os.path.exists(os.path.join(out_dir, f"graph_{i}.pt"))
                   for i in range(len(self.dates) - window)):
            self._build_graphs(out_dir)

    def __len__(self):
        return len(self.dates) - self.window

    def __getitem__(self, idx):
        path = os.path.join(
            self.dest,
            f"{self.market}_{self.mode}_{self.start.date()}_{self.end.date()}_{self.window}",
            f"graph_{idx}.pt"
        )
        return torch.load(path)

    @staticmethod
    def _entropy(arr: np.ndarray) -> float:
        vals, counts = np.unique(arr, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log(p + 1e-12))

    def _adjacency(self, window_slice: np.ndarray) -> torch.Tensor:
        if self.fast_approx:
            t = 5
            A_tilde = A + np.eye(N)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-12))
            H = D_inv_sqrt @ A @ D_inv_sqrt
            A = expm(-t * (np.eye(N) - H))
            return torch.from_numpy(A.astype(np.float32))
        else:
            T = self.window
            N = window_slice.shape[1]
            # Flatten per-node: log1p, reorder to (N, T*F)
            X = window_slice.transpose(1, 0, 2).reshape(N, -1)
            X = np.log1p(X)
        
            energy = np.einsum('ij,ij->i', X, X)
            entropy = np.apply_along_axis(self._entropy, 1, X)
            e_ratio = energy[:, None] / (energy[None, :] + 1e-12)
            ent_sum = entropy[:, None] + entropy[None, :]
        
            # Joint entropy
            X_pair = np.concatenate([
                X[:, None, :].repeat(N, axis=1),
                X[None, :, :].repeat(N, axis=0)
            ], axis=-1)
            joint_ent = np.apply_along_axis(self._entropy, 2, X_pair)
        
            A = e_ratio * np.exp(ent_sum - joint_ent)
            A = np.maximum(A, 1.0)

        return torch.from_numpy(np.log(A).astype(np.float32))

    def _build_graphs(self, out_dir: str):
        n = len(self.dates)
        N = len(self.tickers)
        for i in range(n - self.window):
            idxs = [self.dates.index(d) for d in self.dates[i:i + self.window + 1]]
            data_slice = self.features[idxs]  # shape (T+1, N, F)

            # Labels: count of days close price rose
            closes = data_slice[:, :, 3]
            Y = (closes[1:] > closes[:-1]).sum(axis=0).astype(np.int64)

            A = self._adjacency(data_slice[:-1])
            W = data_slice[:-1]
            X = torch.from_numpy(np.log1p(W.transpose(1, 0, 2).reshape(N, -1)).astype(np.float32))

            torch.save({'X': X, 'A': A, 'Y': torch.from_numpy(Y)},
                       os.path.join(out_dir, f"graph_{i}.pt"))
