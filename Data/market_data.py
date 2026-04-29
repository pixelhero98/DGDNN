"""Market data helpers for current S&P 500 experiments."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

SP500_CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DEFAULT_USER_AGENT = (
    "DGDNN open-source data fetcher "
    "(https://github.com/pixelhero98/DGDNN; research use)"
)

__all__ = [
    "DownloadResult",
    "OHLCV_COLUMNS",
    "SP500_CONSTITUENTS_URL",
    "download_ohlcv",
    "fetch_sp500_constituents",
    "list_missing_ohlcv_files",
    "list_ohlcv_coverage_gaps",
    "normalize_yahoo_symbol",
    "select_sp500_tickers",
]


@dataclass(frozen=True)
class DownloadResult:
    """Summary of an OHLCV download pass."""

    tickers: List[str]
    metadata_path: Path
    downloaded: List[Dict[str, object]]
    failures: List[Dict[str, str]]


def normalize_yahoo_symbol(symbol: str) -> str:
    """Convert an exchange-style symbol into the form expected by Yahoo Finance."""

    return symbol.strip().upper().replace(".", "-")


def fetch_sp500_constituents(
    url: str = SP500_CONSTITUENTS_URL,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
) -> pd.DataFrame:
    """Fetch the current S&P 500 constituents table from Wikipedia."""

    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text), attrs={"id": "constituents"})
    if not tables:
        raise ValueError("Could not find the S&P 500 constituents table")

    frame = tables[0].copy()
    required = {"Symbol", "Security"}
    missing = required - set(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"S&P 500 table is missing required columns: {names}")

    frame["Symbol"] = frame["Symbol"].astype(str).str.strip().str.upper()
    frame["Yahoo Symbol"] = frame["Symbol"].map(normalize_yahoo_symbol)
    return frame


def select_sp500_tickers(
    constituents: pd.DataFrame,
    requested: Optional[Sequence[str]] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """Select Yahoo-ready S&P 500 tickers from a current constituents table."""

    if "Symbol" not in constituents or "Yahoo Symbol" not in constituents:
        raise ValueError("constituents must contain Symbol and Yahoo Symbol columns")

    table = constituents.copy()
    table["Symbol"] = table["Symbol"].astype(str).str.strip().str.upper()
    table["Yahoo Symbol"] = table["Yahoo Symbol"].astype(str).map(normalize_yahoo_symbol)

    if not requested:
        selected = table.reset_index(drop=True)
        return selected["Yahoo Symbol"].tolist(), selected

    lookup: Dict[str, int] = {}
    for index, row in table.iterrows():
        original = str(row["Symbol"]).strip().upper()
        yahoo = str(row["Yahoo Symbol"]).strip().upper()
        lookup[original] = index
        lookup[yahoo] = index
        lookup[normalize_yahoo_symbol(original)] = index

    selected_indices: List[int] = []
    invalid: List[str] = []
    seen: set[int] = set()
    for symbol in requested:
        key = str(symbol).strip().upper()
        if not key:
            continue
        index = lookup[key] if key in lookup else lookup.get(normalize_yahoo_symbol(key))
        if index is None:
            invalid.append(symbol)
            continue
        if index not in seen:
            selected_indices.append(index)
            seen.add(index)

    if invalid:
        bad = ", ".join(invalid)
        raise ValueError(f"Ticker(s) are not current S&P 500 constituents: {bad}")
    if not selected_indices:
        raise ValueError("No usable S&P 500 tickers were selected")

    selected = table.loc[selected_indices].reset_index(drop=True)
    return selected["Yahoo Symbol"].tolist(), selected


def list_missing_ohlcv_files(root: Path, market: str, tickers: Iterable[str]) -> List[str]:
    """Return tickers whose repo-compatible OHLCV CSV is missing."""

    root = Path(root)
    missing = []
    for ticker in tickers:
        path = root / f"{market}_{ticker}_30Y.csv"
        if not path.exists():
            missing.append(ticker)
    return missing


def list_ohlcv_coverage_gaps(
    root: Path,
    market: str,
    tickers: Iterable[str],
    start: str,
    end: str,
) -> List[str]:
    """Return tickers whose CSV is missing, malformed, or outside date coverage."""

    root = Path(root)
    start_ts = pd.to_datetime(start).normalize()
    end_ts = pd.to_datetime(end).normalize()
    gaps = []
    for ticker in tickers:
        path = root / f"{market}_{ticker}_30Y.csv"
        if not path.exists():
            gaps.append(ticker)
            continue
        try:
            frame = pd.read_csv(path, parse_dates=[0], index_col=0)
            missing_columns = [column for column in OHLCV_COLUMNS if column not in frame.columns]
            if missing_columns:
                gaps.append(ticker)
                continue
            frame.index = pd.to_datetime(frame.index).normalize()
            if frame.empty or frame.index.min() > start_ts or frame.index.max() < end_ts:
                gaps.append(ticker)
        except Exception:  # noqa: BLE001 - caller only needs to know it must refetch
            gaps.append(ticker)
    return gaps


def download_ohlcv(
    tickers: Sequence[str],
    root: Path,
    start: str,
    end: str,
    market: str = "SP500",
    auto_adjust: bool = False,
    metadata: Optional[pd.DataFrame] = None,
    on_failure: str = "raise",
) -> DownloadResult:
    """Download Yahoo Finance OHLCV data and write repo-compatible CSV files.

    ``end`` is treated as inclusive by this helper, even though yfinance uses an
    exclusive end date.
    """

    if on_failure not in {"raise", "drop"}:
        raise ValueError("on_failure must be 'raise' or 'drop'")

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised in real installs
        raise ImportError("Install yfinance to download fresh OHLCV data") from exc

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    end_exclusive = (
        pd.to_datetime(end).normalize() + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    start_date = pd.to_datetime(start).normalize().strftime("%Y-%m-%d")

    metadata_by_yahoo: Dict[str, Dict[str, object]] = {}
    if metadata is not None and "Yahoo Symbol" in metadata:
        for _, row in metadata.iterrows():
            metadata_by_yahoo[normalize_yahoo_symbol(str(row["Yahoo Symbol"]))] = row.to_dict()

    downloaded: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    usable: List[str] = []

    for ticker in tickers:
        yahoo_symbol = normalize_yahoo_symbol(ticker)
        try:
            frame = yf.download(
                yahoo_symbol,
                start=start_date,
                end=end_exclusive,
                auto_adjust=auto_adjust,
                progress=False,
                actions=False,
                threads=False,
            )
            frame = _normalize_yfinance_frame(frame, yahoo_symbol)
            if frame.empty:
                raise ValueError("Yahoo returned no OHLCV rows")

            path = root / f"{market}_{yahoo_symbol}_30Y.csv"
            frame.to_csv(path, index_label="Date")
            first_date = frame.index.min().date().isoformat()
            last_date = frame.index.max().date().isoformat()
            record = {
                "symbol": metadata_by_yahoo.get(yahoo_symbol, {}).get("Symbol", yahoo_symbol),
                "yahoo_symbol": yahoo_symbol,
                "csv_path": str(path),
                "rows": int(len(frame)),
                "first_date": first_date,
                "last_date": last_date,
            }
            downloaded.append(record)
            usable.append(yahoo_symbol)
        except Exception as exc:  # noqa: BLE001 - keep downloading other tickers
            failures.append({"ticker": yahoo_symbol, "error": str(exc)})

    metadata_path = root / "download_metadata.json"
    payload = {
        "market": market,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": start_date,
        "end_inclusive": pd.to_datetime(end).normalize().strftime("%Y-%m-%d"),
        "auto_adjust": auto_adjust,
        "requested_tickers": [normalize_yahoo_symbol(ticker) for ticker in tickers],
        "usable_tickers": usable,
        "downloaded": downloaded,
        "failures": failures,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if failures and on_failure == "raise":
        failed = ", ".join(item["ticker"] for item in failures)
        raise RuntimeError(
            f"Failed to download OHLCV data for: {failed}. "
            f"Details were written to {metadata_path}"
        )

    return DownloadResult(
        tickers=usable,
        metadata_path=metadata_path,
        downloaded=downloaded,
        failures=failures,
    )


def _normalize_yfinance_frame(frame: pd.DataFrame, yahoo_symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized = _collapse_yfinance_multiindex(normalized, yahoo_symbol)

    missing = [column for column in OHLCV_COLUMNS if column not in normalized.columns]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Yahoo response is missing columns: {names}")

    normalized = normalized[OHLCV_COLUMNS].copy()
    normalized.index = pd.to_datetime(normalized.index).tz_localize(None).normalize()
    normalized.sort_index(inplace=True)
    normalized = normalized.apply(pd.to_numeric, errors="coerce")
    normalized.dropna(subset=OHLCV_COLUMNS, inplace=True)
    return normalized


def _collapse_yfinance_multiindex(frame: pd.DataFrame, yahoo_symbol: str) -> pd.DataFrame:
    for level in range(frame.columns.nlevels):
        values = {str(value).upper() for value in frame.columns.get_level_values(level)}
        if yahoo_symbol.upper() in values:
            return frame.xs(yahoo_symbol, axis=1, level=level, drop_level=True)

    for level in range(frame.columns.nlevels):
        values = set(frame.columns.get_level_values(level))
        if len(values) == 1:
            return frame.droplevel(level, axis=1)

    return frame
