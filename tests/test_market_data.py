from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from Data.market_data import (
    download_ohlcv,
    fetch_sp500_constituents,
    normalize_yahoo_symbol,
    select_sp500_tickers,
)


def test_fetch_and_select_sp500_constituents(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <table id="constituents">
      <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
      <tr><td>MMM</td><td>3M</td><td>Industrials</td></tr>
      <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>
    </table>
    """

    class Response:
        text = html

        @staticmethod
        def raise_for_status() -> None:
            return None

    def fake_get(*args: object, **kwargs: object) -> Response:
        assert "headers" in kwargs
        return Response()

    monkeypatch.setattr("requests.get", fake_get)

    constituents = fetch_sp500_constituents()
    assert constituents["Yahoo Symbol"].tolist() == ["MMM", "BRK-B"]

    selected, metadata = select_sp500_tickers(constituents, ["brk-b"])
    assert selected == ["BRK-B"]
    assert metadata.loc[0, "Symbol"] == "BRK.B"


def test_download_ohlcv_writes_repo_compatible_csv(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_download(
        ticker: str,
        start: str,
        end: str,
        auto_adjust: bool,
        progress: bool,
        actions: bool,
        threads: bool,
    ) -> pd.DataFrame:
        assert ticker == "BRK-B"
        assert auto_adjust is False
        index = pd.bdate_range(start, periods=3)
        return pd.DataFrame(
            {
                "Open": [1.0, 2.0, 3.0],
                "High": [1.5, 2.5, 3.5],
                "Low": [0.5, 1.5, 2.5],
                "Close": [1.2, 2.2, 3.2],
                "Adj Close": [1.2, 2.2, 3.2],
                "Volume": [100, 110, 120],
            },
            index=index,
        )

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))

    result = download_ohlcv(["BRK.B"], tmp_path, "2023-01-01", "2023-01-10")
    csv_path = tmp_path / "SP500_BRK-B_30Y.csv"
    assert result.tickers == ["BRK-B"]
    assert result.failures == []
    assert csv_path.exists()
    assert result.metadata_path.exists()

    frame = pd.read_csv(csv_path)
    assert frame.columns.tolist() == ["Date", "Open", "High", "Low", "Close", "Volume"]


def test_normalize_yahoo_symbol() -> None:
    assert normalize_yahoo_symbol(" brk.b ") == "BRK-B"
