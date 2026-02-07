"""
Data loading helpers for stage-two pattern clustering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from multi_agent_stock_selection.config.config import GlobalConfig
from multi_agent_stock_selection.patterns.models import Thesis

logger = logging.getLogger(__name__)

ANALYST_POOL_REDIRECTION = {
    "sse_50": "csi_300",  # SSE 50 arguments stored under CSI 300 folder
}


@dataclass(frozen=True, slots=True)
class PoolSnapshot:
    date: int
    stocks: List[str]


class PatternDataLoader:
    """Loads pool membership, thesis JSON files and return labels."""

    def __init__(
        self,
        config: GlobalConfig,
        pool_name: str,
        start_date: int,
        end_date: int,
        analyst_types: Sequence[str],
        return_column: str = "5_15_labelB",
    ) -> None:
        self.config = config
        self.pool_name = pool_name
        self.analysis_pool_name = ANALYST_POOL_REDIRECTION.get(pool_name, pool_name)
        self.start_date = start_date
        self.end_date = end_date
        self.return_column = return_column
        self.analyst_types = list(analyst_types)
        self.analysis_root = Path(config.data.analysis_res_path)

        self.stock_pool = self._load_stock_pool()
        self.pool_by_date = self._split_pool_by_date()

        self.stock_returns = self._load_stock_returns()
        self.excess_return_lookup = self._compute_excess_return_lookup()

    def _load_stock_pool(self) -> pd.DataFrame:
        pool_path = Path(self.config.data.data_root) / f"{self.pool_name}.parq"
        if not pool_path.exists():
            raise FileNotFoundError(f"Stock pool file not found: {pool_path}")

        df = pd.read_parquet(pool_path, columns=["Date", "Stock"])
        df = df[(df["Date"] >= self.start_date) & (df["Date"] <= self.end_date)].copy()
        if df.empty:
            raise ValueError(f"No pool constituents for {self.pool_name} between {self.start_date} and {self.end_date}.")

        # df["Stock"] = df["Stock"].astype(str).str.zfill(6)
        df = df.drop_duplicates(subset=["Date", "Stock"]).sort_values(["Date", "Stock"]).reset_index(drop=True)
        return df

    def _split_pool_by_date(self) -> Dict[int, PoolSnapshot]:
        pool_by_date: Dict[int, PoolSnapshot] = {}
        for date, group in self.stock_pool.groupby("Date"):
            pool_by_date[int(date)] = PoolSnapshot(date=int(date), stocks=group["Stock"].tolist())
        return pool_by_date

    def _load_stock_returns(self) -> pd.DataFrame:
        labels_path = Path(self.config.data.stock_labels_path)
        if not labels_path.exists():
            raise FileNotFoundError(f"Stock label file not found: {labels_path}")

        # columns = ["Stock", "Date", self.return_column]
        df = pd.read_parquet(labels_path)
        if self.return_column not in df.columns:
            raise ValueError(f"Return column {self.return_column} missing in {labels_path}.")

        df = df[(df["Date"] >= self.start_date) & (df["Date"] <= self.end_date)].copy()
        # df["Stock"] = df["Stock"].astype(str).str.zfill(6)
        return df.rename(columns={self.return_column: "ret"})

    def _compute_excess_return_lookup(self) -> Dict[tuple[str, int], float]:
        merged = self.stock_returns.merge(self.stock_pool[["Stock", "Date"]], on=["Stock", "Date"], how="inner")
        if merged.empty:
            logger.warning("Merged pool/returns frame is empty, excess returns will be unavailable.")
            return {}
        # label_ret
        merged["avg_ret"] = merged.groupby("Date")["label_ret"].transform("mean")
        merged["excess_ret"] = merged["label_ret"] - merged["avg_ret"]
        lookup = {(row.Stock, int(row.Date)): float(row.excess_ret) for row in merged.itertuples()}
        return lookup

    @property
    def trading_dates(self) -> List[int]:
        dates = sorted(self.pool_by_date.keys())
        return [date for date in dates if self.start_date <= date <= self.end_date]

    def get_pool_stocks(self, date: int) -> List[str]:
        snapshot = self.pool_by_date.get(date)
        return snapshot.stocks if snapshot else []

    def get_pool_frame(self, date: int) -> pd.DataFrame:
        stocks = self.get_pool_stocks(date)
        if not stocks:
            return pd.DataFrame(columns=["Stock"])
        return pd.DataFrame({"Stock": stocks})

    def load_theses_for_date(self, date: int) -> List[Thesis]:
        stocks = self.get_pool_stocks(date)
        theses: List[Thesis] = []
        if not stocks:
            logger.warning("No stocks found for pool %s on %s", self.pool_name, date)
            return theses

        for stock in stocks:
            for analyst in self.analyst_types:
                path = self._resolve_analysis_path(analyst, stock, date)
                if not path.exists():
                    continue
                try:
                    with path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", path, exc)
                    continue
                entries = self._normalize_entries(payload)
                for entry in entries:
                    content = self._extract_content(entry)
                    if not content:
                        logger.warning(f"No content found in thesis entry: {entry} on stock {stock}, date {date}, analyst {analyst}")
                        continue
                    polarity = self._extract_polarity(entry)
                    if polarity is None:
                        continue
                    theses.append(
                        Thesis(
                            stock=stock,
                            date=date,
                            analyst=analyst,
                            polarity=polarity,
                            content=content,
                            extra={"source_path": str(path)},
                        )
                    )
        return theses

    def _resolve_analysis_path(self, analyst: str, stock: str, date: int) -> Path:
        file_name = f"{stock}_{date}.json"
        return self.analysis_root / analyst / self.analysis_pool_name / file_name

    @staticmethod
    def _normalize_entries(payload: dict) -> Iterable[dict]:
        if not isinstance(payload, dict):
            return []
        if isinstance(payload.get("analysis_content"), list):
            return payload["analysis_content"]
        if isinstance(payload.get("arguments"), list):
            return payload["arguments"]
        return []

    @staticmethod
    def _extract_content(entry: dict) -> str | None:
        for key in ("thesis_content", "theis_content", "argument_content", "content"):
            value = entry.get(key)
            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value
        return None

    @staticmethod
    def _extract_polarity(entry: dict) -> bool | None:
        value = entry.get("thesis_polar") or entry.get("polar") or entry.get("thesis_polarity")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in {"bullish", "positive", "long", "true"}:
                return True
            if value_lower in {"bearish", "negative", "short", "false"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None
