"""Training and evaluation script for next-day stock movement prediction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from Data.dataset_gen import MyDataset
from Model.dgdnn import DGDNN


@dataclass(frozen=True)
class DateRange:
    """Simple container storing the start and end timestamps for a dataset."""

    start: str
    end: str


@dataclass
class ExperimentConfig:
    """Configuration values for reproducing the experiment."""

    market: str = "NASDAQ"
    window: int = 19
    fast_approx: bool = False
    epochs: int = 6000
    print_every: int = 100
    train_range: DateRange = DateRange("2013-01-01", "2014-12-31")
    val_range: DateRange = DateRange("2015-01-01", "2015-06-30")
    test_range: DateRange = DateRange("2015-07-01", "2017-12-31")


def load_tickers(csv_path: Path) -> List[str]:
    """Read a single-column CSV file containing ticker symbols."""

    tickers: List[str] = []
    with csv_path.open(newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if row:
                tickers.append(row[0].strip())
    return tickers


def build_datasets(
    config: ExperimentConfig,
    tickers: Sequence[str],
    data_root: Path,
    cache_dir: Path,
) -> Tuple[MyDataset, MyDataset, MyDataset]:
    """Construct train/validation/test datasets with shared parameters."""

    kwargs = {
        "root": str(data_root),
        "dest": str(cache_dir),
        "market": config.market,
        "tickers": tickers,
        "window": config.window,
        "fast_approx": config.fast_approx,
    }
    train_ds = MyDataset(**kwargs, start=config.train_range.start, end=config.train_range.end, mode="train")
    val_ds = MyDataset(**kwargs, start=config.val_range.start, end=config.val_range.end, mode="validation")
    test_ds = MyDataset(**kwargs, start=config.test_range.start, end=config.test_range.end, mode="test")
    return train_ds, val_ds, test_ds


def initialize_model(node_count: int, config: ExperimentConfig) -> DGDNN:
    """Create the DGDNN model using the default hyper-parameters from the paper."""

    layers, expansion_step, num_heads = 6, 7, 2
    classes = 2
    emb_hidden_size, emb_output_size, raw_feature_size = 1024, 256, 64
    timestamp = config.window

    diffusion_size = [timestamp * len(MyDataset.FEATURE_COLUMNS), 64, 128, 256, 256, 256, 128]
    emb_size = [128, 384, 512, 512, 512, 384]

    if num_heads != 2:
        scale = num_heads / 2.0
        emb_output_size = int(round(emb_output_size * scale))
        raw_feature_size = int(round(raw_feature_size * scale))
        diffusion_size = [diffusion_size[0]] + [int(round(dim * scale)) for dim in diffusion_size[1:]]
        emb_size = [int(round(dim * scale)) for dim in emb_size]

    return DGDNN(
        diffusion_size,
        emb_size,
        emb_hidden_size,
        emb_output_size,
        raw_feature_size,
        classes,
        layers,
        node_count,
        expansion_step,
        num_heads,
        active=[True] * layers,
    )


def train(
    model: DGDNN,
    optimizer: torch.optim.Optimizer,
    dataset: MyDataset,
    device: torch.device,
    epochs: int,
    print_every: int,
) -> None:
    """Run a simple supervised training loop."""

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_examples = 0.0, 0, 0
        for sample in dataset:
            inputs = sample["X"].to(device)
            adjacency = sample["A"].to(device)
            targets = sample["Y"].to(device).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, adjacency)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += int((logits.argmax(dim=1) == targets).sum())
            total_examples += targets.size(0)

        if epoch % print_every == 0:
            accuracy = total_correct / max(total_examples, 1)
            print(f"Epoch {epoch:05d} | loss={total_loss:.4f} | acc={accuracy:.4f}")


@torch.no_grad()
def evaluate(model: DGDNN, dataset: MyDataset, device: torch.device, name: str) -> Dict[str, float]:
    """Compute Accuracy, macro-F1 and MCC metrics on a dataset."""

    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    for sample in dataset:
        inputs = sample["X"].to(device)
        adjacency = sample["A"].to(device)
        targets = sample["Y"].to(device)
        logits = model(inputs, adjacency)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(all_targets, all_preds),
        "f1_macro": f1_score(all_targets, all_preds, average="macro"),
        "mcc": matthews_corrcoef(all_targets, all_preds),
    }
    print(
        f"{name:>10s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f} | MCC: {metrics['mcc']:.4f}"
    )
    return metrics


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "Data"
    cache_dir = data_root / "processed_graphs"

    config = ExperimentConfig()
    ticker_csv = data_root / f"{config.market}.csv"
    if not ticker_csv.exists():
        raise FileNotFoundError(f"Ticker list CSV not found at {ticker_csv}")

    tickers = load_tickers(ticker_csv)
    if not tickers:
        raise ValueError("Ticker list is empty; please populate the CSV file with symbols.")

    train_ds, val_ds, test_ds = build_datasets(config, tickers, data_root, cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(len(tickers), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1.5e-5)

    print("Starting training ...")
    train(model, optimizer, train_ds, device, config.epochs, config.print_every)
    print("Training complete. Evaluating ...")
    evaluate(model, val_ds, device, "Validation")
    evaluate(model, test_ds, device, "Test")


if __name__ == "__main__":
    main()

