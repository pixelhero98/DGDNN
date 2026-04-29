from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Data.dataset_gen import DatasetConfig, MyDataset


TICKERS = ["AAA", "BBB", "CCC"]
MARKET = "SP500"


def write_synthetic_csvs(root: Path, start: str = "2022-12-29") -> pd.DatetimeIndex:
    dates = pd.bdate_range(start, periods=12)
    root.mkdir(parents=True, exist_ok=True)
    for offset, ticker in enumerate(TICKERS):
        base = 10.0 + offset * 5.0
        rows = []
        for index, _date in enumerate(dates):
            value = base + index
            rows.append(
                {
                    "Open": value,
                    "High": value + 1.0,
                    "Low": value - 1.0,
                    "Close": value + 0.5,
                    "Volume": 1000.0 + offset * 100.0 + index * 10.0,
                }
            )
        frame = pd.DataFrame(rows, index=dates)
        frame.to_csv(root / f"{MARKET}_{ticker}_30Y.csv", index_label="Date")
    return dates


def build_config(root: Path, dest: Path, **overrides: object) -> DatasetConfig:
    values = {
        "root": str(root),
        "dest": str(dest),
        "market": MARKET,
        "tickers": TICKERS,
        "start": "2023-01-02",
        "end": "2023-01-11",
        "window": 3,
        "mode": "train",
    }
    values.update(overrides)
    return DatasetConfig(**values)


def test_ratio_is_default_and_matches_raw_arithmetic(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    dest = tmp_path / "graphs"
    write_synthetic_csvs(root)

    dataset = MyDataset(build_config(root, dest))
    sample = dataset[0]
    assert sample["X"].shape == (3, 15)
    assert sample["A"].shape == (3, 3)
    assert sample["Y"].shape == (3,)

    frame = pd.read_csv(root / "SP500_AAA_30Y.csv", parse_dates=[0], index_col=0)
    current = frame.loc[pd.Timestamp("2023-01-02"), ["Open", "High", "Low", "Close", "Volume"]]
    previous = frame.loc[pd.Timestamp("2022-12-30"), ["Open", "High", "Low", "Close", "Volume"]]
    expected = (current.to_numpy(dtype=np.float64) - previous.to_numpy(dtype=np.float64)) / np.maximum(
        np.abs(previous.to_numpy(dtype=np.float64)),
        1e-6,
    )
    np.testing.assert_allclose(sample["X"][0, :5].numpy(), expected, rtol=1e-6)

    manifest_path = Path(dataset.output_directory) / "graph_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["fingerprint"]["feature_transform"] == "ratio"


def test_log_feature_switch_and_cache_invalidation(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    dest = tmp_path / "graphs"
    write_synthetic_csvs(root)

    ratio_dataset = MyDataset(build_config(root, dest))
    ratio_sample = ratio_dataset[0]["X"].clone()

    log_dataset = MyDataset(build_config(root, dest, feature_transform="log"))
    log_sample = log_dataset[0]["X"]
    assert not np.allclose(ratio_sample.numpy(), log_sample.numpy())

    frame = pd.read_csv(root / "SP500_AAA_30Y.csv", parse_dates=[0], index_col=0)
    raw = frame.loc[pd.Timestamp("2023-01-02"), ["Open", "High", "Low", "Close", "Volume"]]
    expected = np.log1p(raw.to_numpy(dtype=np.float64) + 1e-6)
    np.testing.assert_allclose(log_sample[0, :5].numpy(), expected, rtol=1e-6)

    manifest = json.loads(
        (Path(log_dataset.output_directory) / "graph_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["fingerprint"]["feature_transform"] == "log"


def test_positive_adjacency_filter_removes_non_positive_off_diagonal(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    dest = tmp_path / "graphs"
    write_synthetic_csvs(root)

    dataset = MyDataset(build_config(root, dest, positive_adj_filter=True))
    adjacency = dataset[0]["A"].numpy()
    off_diagonal = ~np.eye(adjacency.shape[0], dtype=bool)
    assert np.all(adjacency[off_diagonal] >= 0.0)
    assert np.allclose(np.diag(adjacency), 0.0)


def test_ratio_transform_requires_lookback_rows(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    dest = tmp_path / "graphs"
    write_synthetic_csvs(root, start="2023-01-02")

    with pytest.raises(ValueError, match="include lookback rows"):
        MyDataset(build_config(root, dest))
