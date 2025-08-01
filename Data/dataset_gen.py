import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.linalg import expm


class MyDataset(Dataset):
    """
    A PyTorch Dataset for rolling-window graphs of company time series.

    Parameters:
      - root (str): directory containing CSVs named {market}_{ticker}_30Y.csv
      - dest (str): output directory for serialized graphs
      - market (str): market code prefix in filenames (e.g., 'NASDAQ')
      - tickers (list[str]): list of ticker symbols to include as nodes (e.g., ['AAPL','MSFT'])
      - start (str): inclusive window start date 'YYYY-MM-DD'
      - end (str): inclusive window end date 'YYYY-MM-DD'
      - window (int): number of past days T to use per graph
      - mode (str, optional): subfolder label, e.g. 'train' or 'test' (default 'train')
      - fast_approx (bool, optional): whether to use heat-kernel approximation (default False)
      - heat_tau (float, optional): time parameter for heat kernel (default 5.0)
      - sparsify_threshold (float, optional): threshold for sparsification (default 0.3)
      - log_eps (float, optional): epsilon added before log (default 1e-12)
      - norm_eps (float, optional): epsilon added for numeric stability in z-score (default 1e-6)

    Graph dict entries:
      - X: Tensor [N, F * T] node features (log1p normalized)
      - A: Tensor [N, N] adjacency matrix (entropy-energy or heat-kernel)
      - Y: Tensor [N] integer labels (days price rose in window)
    """
    def __init__(
        self,
        root: str,
        dest: str,
        market: str,
        tickers: list[str],
        start: str,
        end: str,
        window: int,
        mode: str = 'train',
        fast_approx: bool = False,
        heat_tau: float = 5.0,
        sparsify_threshold: float = 0.3,
        log_eps: float = 1e-12,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.root = root
        self.dest = dest
        self.market = market
        self.tickers = tickers
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.window = window
        self.mode = mode
        self.fast_approx = fast_approx
        self.heat_tau = heat_tau
        self.sparsify_threshold = sparsify_threshold
        self.log_eps = log_eps
        self.norm_eps = norm_eps

        # Load data_frames_full & in-range slice
        self.data_frames_full = {}
        self.data_frames = {}
        for t in self.tickers:
            path = os.path.join(root, f"{market}_{t}_30Y.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV file for ticker {t} not found at {path}")
            df = pd.read_csv(path, parse_dates=[0], index_col=0)
            self.data_frames_full[t] = df
            self.data_frames[t] = df.loc[self.start:self.end]

        # Common trading dates
        common = set.intersection(*[set(df.index.normalize()) for df in self.data_frames.values()])
        self.dates = sorted(common)

        # Next common date after end for labeling
        after_sets = [set(df.index.normalize()[df.index.normalize() > self.end]) for df in self.data_frames_full.values()]
        common_after = set.intersection(*after_sets)
        self.next_day = min(common_after) if common_after else None

        # Stack raw features: shape (n_dates, N, F)
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.features = np.stack([self.data_frames[t][feature_cols].values for t in self.tickers], axis=1)

        # Prepare output
        out_dir = os.path.join(dest, f"{market}_{mode}_{self.start.date()}_{self.end.date()}_{window}")
        os.makedirs(out_dir, exist_ok=True)

        # Build if missing
        total = len(self.dates) - window + (1 if self.next_day else 0)
        if not all(os.path.exists(os.path.join(out_dir, f"graph_{i}.pt")) for i in range(total)):
            self._build_graphs(out_dir)

    def __len__(self):
        return len(self.dates) - self.window + (1 if self.next_day else 0)

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

    def _adjacency(self, X: np.ndarray) -> torch.Tensor:
        """
        Build adjacency from precomputed feature matrix X (N, T*F).
        Uses entropy-energy and optional heat diffusion.
        """
        N = X.shape[0]
        energy = np.einsum('ij,ij->i', X, X)
        entropy = np.apply_along_axis(self._entropy, 1, X)
        e_ratio = energy[:, None] / (energy[None, :] + self.log_eps)
        ent_sum = entropy[:, None] + entropy[None, :]

        # joint entropy
        X_pair = np.concatenate([
            X[:, None, :].repeat(N, axis=1),
            X[None, :, :].repeat(N, axis=0)
        ], axis=-1)
        joint_ent = np.apply_along_axis(self._entropy, 2, X_pair)

        A = e_ratio * (np.exp(ent_sum - joint_ent) - 1)

        if self.fast_approx:
            A_t = A + np.eye(N)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(A_t.sum(axis=1) + self.log_eps))
            H = D_inv_sqrt @ A_t @ D_inv_sqrt
            A = expm(-self.heat_tau * (np.eye(N) - H))
        else:
            A[A < self.sparsify_threshold] = 0.0
            A = np.log(A + self.log_eps)

        A = (A + A.T) / 2.0
        np.fill_diagonal(A, 0.0)
        return torch.from_numpy(A.astype(np.float32))

    def _build_graphs(self, out_dir: str):
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        n = len(self.dates)
        for i in range(n - self.window + (1 if self.next_day else 0)):
            # select dates
            if i < len(self.dates) - self.window:
                slice_dates = self.dates[i:i+self.window+1]
            else:
                slice_dates = self.dates[-self.window:] + [self.next_day]

            # collect slices
            data = []
            for d in slice_dates:
                if d in self.dates:
                    idx = self.dates.index(d)
                    data.append(self.features[idx])
                else:
                    rows = [self.data_frames_full[t].loc[d, feature_cols].values for t in self.tickers]
                    data.append(np.stack(rows, axis=0))
            slice_arr = np.stack(data, axis=0)  # (T+1, N, F)

            # label
            closes = slice_arr[:,:,3]
            Y = (closes[-1]>closes[-2]).astype(np.int64)

            # features: log1p
            W = slice_arr[:-1]
            N = W.shape[1]
            X_norm = np.log1p(W.transpose(1,0,2).reshape(N,-1))

            # adjacency from normalized X
            A = self._adjacency(X_norm)
            X = torch.from_numpy(X_norm.astype(np.float32))

            torch.save({'X':X,'A':A,'Y':torch.from_numpy(Y)}, os.path.join(out_dir,f"graph_{i}.pt"))

