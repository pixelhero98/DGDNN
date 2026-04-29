"""Train and evaluate the Dynamic Graph Diffusion Neural Network."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data.dataset_gen import DatasetConfig, MyDataset
from Data.market_data import (
    download_ohlcv,
    fetch_sp500_constituents,
    list_ohlcv_coverage_gaps,
    normalize_yahoo_symbol,
    select_sp500_tickers,
)
from Model.dgdnn import DGDNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2024-12-31"


@dataclass(frozen=True)
class ExperimentConfig:
    """All hyper-parameters required for training the model."""

    root: Path
    destination: Path
    market: str
    tickers: Sequence[str]
    train_range: Tuple[str, str]
    val_range: Tuple[str, str]
    test_range: Tuple[str, str]
    window: int = 19
    fast_approximation: bool = False
    layers: int = 6
    expansion_step: int = 7
    num_heads: int = 2
    embedding_hidden: int = 1024
    embedding_output: int = 256
    raw_feature_size: int = 64
    classes: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1.5e-5
    epochs: int = 6000
    feature_transform: str = "ratio"
    positive_adj_filter: bool = False


def read_ticker_file(path: Path) -> List[str]:
    """Load ticker symbols from a CSV file."""

    tickers: List[str] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        tickers = [row[0] for row in reader if row]
    return tickers


def filter_missing(base: Iterable[str], missing: Iterable[str]) -> List[str]:
    """Remove missing tickers while preserving order."""

    missing_set = set(missing)
    return [ticker for ticker in base if ticker not in missing_set]


def build_dataset(config: ExperimentConfig, mode: str, date_range: Tuple[str, str]) -> MyDataset:
    dataset_config = DatasetConfig(
        root=str(config.root),
        dest=str(config.destination),
        market=config.market,
        tickers=list(config.tickers),
        start=date_range[0],
        end=date_range[1],
        window=config.window,
        mode=mode,
        fast_approx=config.fast_approximation,
        feature_transform=config.feature_transform,
        positive_adj_filter=config.positive_adj_filter,
    )
    return MyDataset(dataset_config)


def build_model(config: ExperimentConfig, num_nodes: int, timestamp: int) -> DGDNN:
    if config.layers != 6:
        raise ValueError("build_model currently supports exactly 6 DGDNN layers")

    diffusion_size = [timestamp * 5, 64, 128, 256, 256, 256, 128]
    embedding_size = [64 + 64, 128 + 256, 256 + 256, 256 + 256, 256 + 256, 128 + 256]

    emb_output = config.embedding_output
    raw_feature_size = config.raw_feature_size
    if config.num_heads != 2:
        scale = config.num_heads / 2.0
        emb_output = int(round(emb_output * scale))
        raw_feature_size = int(round(raw_feature_size * scale))
        diffusion_size = [diffusion_size[0]] + [int(round(x * scale)) for x in diffusion_size[1:]]
        embedding_size = [int(round(x * scale)) for x in embedding_size]

    model = DGDNN(
        diffusion_size,
        embedding_size,
        config.embedding_hidden,
        emb_output,
        raw_feature_size,
        config.classes,
        config.layers,
        num_nodes,
        config.expansion_step,
        config.num_heads,
        active=[True] * config.layers,
    )
    return model.to(device)


def train_epoch(model: DGDNN, dataset: MyDataset, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sample in dataset:
        features = sample["X"].to(device)
        adjacency = sample["A"].to(device)
        target = sample["Y"].to(device).long()

        optimizer.zero_grad()
        logits = model(features, adjacency)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        predictions = logits.argmax(dim=1)
        correct += int((predictions == target).sum())
        total += target.size(0)

    accuracy = correct / total if total else 0.0
    return total_loss, accuracy


def evaluate(model: DGDNN, dataset: MyDataset) -> Tuple[float, float, float]:
    model.eval()
    predictions: List[int] = []
    targets: List[int] = []

    with torch.no_grad():
        for sample in dataset:
            features = sample["X"].to(device)
            adjacency = sample["A"].to(device)
            target = sample["Y"].to(device).long()

            logits = model(features, adjacency)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(target.cpu().tolist())

    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    mcc = (
        matthews_corrcoef(targets, predictions)
        if len(set(targets)) > 1 and len(set(predictions)) > 1
        else 0.0
    )
    return accuracy, f1, mcc


def parse_ticker_argument(value: Optional[str]) -> List[str]:
    """Parse a comma-separated ticker argument."""

    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_requested_tickers(tickers: Optional[str], ticker_file: Optional[Path]) -> List[str]:
    """Load requested tickers from CLI options."""

    requested = parse_ticker_argument(tickers)
    if ticker_file is not None:
        requested.extend(read_ticker_file(ticker_file))
    return requested


def common_trading_dates(
    root: Path,
    market: str,
    tickers: Sequence[str],
    start: str,
    end: str,
) -> List[pd.Timestamp]:
    """Find dates available for every ticker inside the requested range."""

    common: Optional[set[pd.Timestamp]] = None
    start_ts = pd.to_datetime(start).normalize()
    end_ts = pd.to_datetime(end).normalize()
    for ticker in tickers:
        path = root / f"{market}_{ticker}_30Y.csv"
        if not path.exists():
            raise FileNotFoundError(f"CSV file for ticker {ticker} not found at {path}")
        frame = pd.read_csv(path, parse_dates=[0], index_col=0)
        frame.index = pd.to_datetime(frame.index).normalize()
        dates = set(frame.loc[start_ts:end_ts].index)
        common = dates if common is None else common & dates

    return sorted(common or set())


def split_common_dates(
    dates: Sequence[pd.Timestamp],
    window: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]]:
    """Split common trading dates into train/validation/test date ranges."""

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")

    total = len(dates)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    if min(train_count, val_count, test_count) <= window:
        raise ValueError(
            "Each split must contain more common trading dates than the model window"
        )

    train_dates = dates[:train_count]
    val_dates = dates[train_count : train_count + val_count]
    test_dates = dates[train_count + val_count :]
    return (
        _date_range(train_dates),
        _date_range(val_dates),
        _date_range(test_dates),
    )


def _date_range(dates: Sequence[pd.Timestamp]) -> Tuple[str, str]:
    return dates[0].date().isoformat(), dates[-1].date().isoformat()


def prepare_market_data(args: argparse.Namespace) -> Tuple[str, List[str]]:
    """Resolve tickers and ensure CSV data exists for the experiment."""

    market = args.market.upper()
    requested = load_requested_tickers(args.tickers, args.ticker_file)
    metadata = None

    if market == "SP500":
        constituents = fetch_sp500_constituents()
        tickers, metadata = select_sp500_tickers(constituents, requested or None)
    else:
        if not requested:
            raise ValueError("Non-SP500 markets require --tickers or --ticker-file")
        tickers = [normalize_yahoo_symbol(ticker) for ticker in requested]

    coverage_start = (
        pd.to_datetime(args.start).normalize() - pd.Timedelta(days=args.lookback_days)
        if args.feature_transform == "ratio"
        else pd.to_datetime(args.start).normalize()
    ).date().isoformat()
    coverage_end = (
        pd.to_datetime(args.end).normalize() + pd.Timedelta(days=args.label_tail_days)
    ).date().isoformat()
    coverage_gaps = list_ohlcv_coverage_gaps(
        args.data_root,
        market,
        tickers,
        start=coverage_start,
        end=coverage_end,
    )
    should_download = args.refresh_data or bool(coverage_gaps)
    if should_download and not args.download:
        missing_text = ", ".join(coverage_gaps) if coverage_gaps else "refresh requested"
        raise FileNotFoundError(
            f"Missing or stale local data ({missing_text}) and --no-download was set"
        )

    if should_download:
        attempted = tickers if args.refresh_data else coverage_gaps
        result = download_ohlcv(
            attempted,
            args.data_root,
            start=coverage_start,
            end=coverage_end,
            market=market,
            metadata=metadata,
            on_failure="drop" if args.drop_failed_downloads else "raise",
        )
        if args.drop_failed_downloads:
            failed = {item["ticker"] for item in result.failures}
            tickers = [ticker for ticker in tickers if ticker not in failed]
            if not tickers:
                raise RuntimeError("All selected tickers failed to download")

    return market, tickers


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DGDNN with current S&P 500 data by default."
    )
    parser.add_argument("--market", default="SP500", help="Market prefix for CSV names")
    parser.add_argument(
        "--tickers",
        help="Comma-separated ticker list. Omit for all current S&P 500 constituents.",
    )
    parser.add_argument("--ticker-file", type=Path, help="CSV file with one ticker per row")
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive experiment start date")
    parser.add_argument("--end", default=DEFAULT_END, help="Inclusive experiment end date")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--graph-root", type=Path, default=ROOT / "data" / "graphs")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch missing OHLCV CSVs with yfinance",
    )
    parser.add_argument("--refresh-data", action="store_true", help="Re-download selected CSVs")
    parser.add_argument(
        "--drop-failed-downloads",
        action="store_true",
        help="Continue after Yahoo download failures by dropping failed tickers",
    )
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--label-tail-days", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--window", type=int, default=19)
    parser.add_argument(
        "--feature-transform",
        choices=("ratio", "log"),
        default="ratio",
        help="Feature transform used before graph construction",
    )
    parser.add_argument(
        "--positive-adj-filter",
        action="store_true",
        help="Zero non-positive post-log off-diagonal adjacency weights",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1.5e-5)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--expansion-step", type=int, default=7)
    parser.add_argument("--num-heads", type=int, default=2)
    return parser.parse_args(argv)


def run_experiment(config: ExperimentConfig) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Build datasets, train a model, and return validation/test metrics."""

    train_dataset = build_dataset(config, "train", config.train_range)
    val_dataset = build_dataset(config, "validation", config.val_range)
    test_dataset = build_dataset(config, "test", config.test_range)

    timestamp = config.window
    model = build_model(config, num_nodes=len(config.tickers), timestamp=timestamp)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    for epoch in range(1, config.epochs + 1):
        loss, acc = train_epoch(model, train_dataset, optimizer)
        print(f"Epoch {epoch:04d}: loss={loss:.4f}, acc={acc:.4f}")

    return evaluate(model, val_dataset), evaluate(model, test_dataset)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for training and evaluation."""

    args = parse_args(argv)
    market, tickers = prepare_market_data(args)
    common_dates = common_trading_dates(args.data_root, market, tickers, args.start, args.end)
    train_range, val_range, test_range = split_common_dates(
        common_dates,
        window=args.window,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    experiment = ExperimentConfig(
        root=args.data_root,
        destination=args.graph_root,
        market=market,
        tickers=tickers,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        window=args.window,
        layers=args.layers,
        expansion_step=args.expansion_step,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        feature_transform=args.feature_transform,
        positive_adj_filter=args.positive_adj_filter,
    )

    print(f"Market: {experiment.market}")
    print(f"Tickers: {len(experiment.tickers)}")
    print(f"Train: {experiment.train_range[0]} to {experiment.train_range[1]}")
    print(f"Validation: {experiment.val_range[0]} to {experiment.val_range[1]}")
    print(f"Test: {experiment.test_range[0]} to {experiment.test_range[1]}")
    print(f"Feature transform: {experiment.feature_transform}")
    print(f"Positive adjacency filter: {experiment.positive_adj_filter}")

    val_metrics, test_metrics = run_experiment(experiment)

    print("Validation -- Acc: {:.4f}, F1: {:.4f}, MCC: {:.4f}".format(*val_metrics))
    print("Test -- Acc: {:.4f}, F1: {:.4f}, MCC: {:.4f}".format(*test_metrics))


if __name__ == "__main__":
    main()
