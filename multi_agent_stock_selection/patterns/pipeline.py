"""
Main orchestration logic for stage-two thought pattern clustering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from multi_agent_stock_selection.config.config import GlobalConfig
from multi_agent_stock_selection.patterns.data_loader import PatternDataLoader
from multi_agent_stock_selection.patterns.embedding import EmbeddingCache, EmbeddingManager, OpenRouterEmbeddingClient
from multi_agent_stock_selection.patterns.models import Thesis, ThesisPatternRecord
from multi_agent_stock_selection.patterns.pattern_state import PatternState

logger = logging.getLogger(__name__)
PATTERN_LOG_DIR = Path("res/pattern_logs")
PATTERN_LOG_DIR.mkdir(parents=True, exist_ok=True)
PATTERN_PLOT_DIR = Path("res/pattern_plots")
PATTERN_PLOT_DIR.mkdir(parents=True, exist_ok=True)

ANALYST_TYPES = ("FundamentalAnalyst", "TechnicalAnalyst", "NewsAnalyst")


class PatternClusteringPipeline:
    """High-level API that wires together the data loader, embeddings, and GMM state."""

    def __init__(
        self,
        config: GlobalConfig,
        pool_name: str,
        start_date: int,
        end_date: int,
        days_per_pattern: int,
        num_patterns: int,
        return_column: str,
        decay_lambda: float,
        embedding_api_key: str,
        beta: float = 1.0,
        distance_metric: str = "euclidean",
        embedding_model: str = "qwen/qwen3-embedding-8b",
        embedding_batch_size: int = 64,
        embedding_cache_dir: Optional[str] = None,
        random_state: int = 42,
        output_path: Optional[str | Path] = None,
    ) -> None:
        if beta < 0:
            raise ValueError("beta must be non-negative.")

        self.config = config
        self.pool_name = pool_name
        self.start_date = start_date
        self.end_date = end_date
        self.days_per_pattern = max(1, days_per_pattern)
        self.num_patterns = num_patterns
        self.decay_lambda = decay_lambda
        self.pipeline_tag = f"{self.pool_name}_{self.num_patterns}_{self.decay_lambda}_{beta}_2025_Q4"

        self.data_loader = PatternDataLoader(
            config=config,
            pool_name=pool_name,
            start_date=start_date,
            end_date=end_date,
            analyst_types=ANALYST_TYPES,
            return_column=return_column,
        )

        cache = EmbeddingCache(embedding_cache_dir) if embedding_cache_dir else None
        embedding_client = OpenRouterEmbeddingClient(
            api_key=embedding_api_key,
            model=embedding_model,
            batch_size=embedding_batch_size,
        )
        self.embedding_manager = EmbeddingManager(embedding_client, cache)

        self.pattern_state = PatternState(
            n_components=num_patterns,
            decay_lambda=decay_lambda,
            random_state=random_state,
            beta=beta,
            distance_metric=distance_metric,
        )

        self.pattern_signal_map: Dict[int, pd.DataFrame] = {}
        self.output_path = Path(output_path) if output_path else None
        self.metrics_log_path = PATTERN_LOG_DIR / f"{self.pipeline_tag}_pattern_metrics.log"
        self.metrics_logger = self._build_metrics_logger()
        self.cluster_plot_dir = PATTERN_PLOT_DIR / self.pipeline_tag
        self.cluster_plot_dir.mkdir(parents=True, exist_ok=True)
        self.ic_history: List[float] = []
        self.rank_ic_history: List[float] = []
        self.selected_dates: List[int] = []

    def _build_metrics_logger(self) -> logging.Logger:
        logger_name = f"{__name__}.metrics.{self.pipeline_tag}"
        metrics_logger = logging.getLogger(logger_name)
        metrics_logger.setLevel(logging.INFO)
        metrics_logger.propagate = False
        if not metrics_logger.handlers:
            self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(self.metrics_log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            metrics_logger.addHandler(handler)
        return metrics_logger

    def _select_pattern_dates(self) -> List[int]:
        dates = self.data_loader.trading_dates
        if not dates:
            return []

        pattern_dates: List[int] = []
        idx = 0
        while idx < len(dates):
            pattern_dates.append(dates[idx])
            idx += self.days_per_pattern
        if pattern_dates[-1] != dates[-1]:
            pattern_dates.append(dates[-1])
        return sorted(set(pattern_dates))

    def run(self) -> pd.DataFrame:
        pattern_dates = self._select_pattern_dates()
        if not pattern_dates:
            logger.warning("No trading dates available for the requested window.")
            return pd.DataFrame(columns=["Date", "Stock", "signal"])

        prev_records: List[ThesisPatternRecord] = []

        for idx, date in enumerate(pattern_dates):
            logger.info("Processing trading day %s (%d/%d)", date, idx + 1, len(pattern_dates))

            if prev_records:
                self.pattern_state.update_performance(prev_records, self.data_loader.excess_return_lookup)

            theses = self.data_loader.load_theses_for_date(date)
            if not theses:
                logger.warning("No theses found on %s, defaulting signals to zero.", date)
                empty_frame = self.data_loader.get_pool_frame(date)
                if not empty_frame.empty:
                    empty_frame["signal"] = 0.0
                    empty_frame.insert(0, "Date", date)
                    self.pattern_signal_map[date] = empty_frame
                    self._compute_signal_quality(date, empty_frame[["Stock", "signal"]].copy())
                prev_records = []
                continue

            embeddings = self._embed_theses(date, theses)
            embedding_matrix = np.vstack(embeddings)

            prev_means = None
            if self.pattern_state.model is not None and getattr(self.pattern_state.model, "means_", None) is not None:
                prev_means = self.pattern_state.model.means_.copy()

            prev_resp = self.pattern_state.predict_proba(embedding_matrix)
            responsibilities, alignment, predicted_prev = self.pattern_state.fit(embedding_matrix, prev_means=prev_means)
            self._assign_thesis_scores(theses, prev_resp, predicted_prev)

            day_signals = self._aggregate_signals(date, theses)
            self.pattern_signal_map[date] = day_signals
            self._compute_signal_quality(date, day_signals[["Stock", "signal"]])

            cluster_labels = np.argmax(responsibilities, axis=1)
            self._log_cluster_quality(date, embedding_matrix, cluster_labels)
            self._save_cluster_plot(date, embedding_matrix, cluster_labels)
            self._log_cluster_migration(date, alignment)
            prev_records = [
                ThesisPatternRecord(
                    stock=thesis.stock,
                    date=thesis.date,
                    polarity_sign=1.0 if thesis.polarity else -1.0,
                    responsibilities=resp,
                )
                for thesis, resp in zip(theses, responsibilities)
            ]

            if self.output_path:
                interim_df = self._forward_fill_signals(end_date=date)
                if not interim_df.empty:
                    self.save(interim_df, self.output_path)

        final_df = self._forward_fill_signals()
        return final_df

    def save(self, df: pd.DataFrame, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            existing = pd.read_parquet(output_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.sort_values(["Date", "Stock"]).drop_duplicates(subset=["Date", "Stock"], keep="last")
        else:
            combined = df.sort_values(["Date", "Stock"])

        combined.to_parquet(output_path, index=False)
        logger.info("Saved %d rows to %s", len(combined), output_path)
        return output_path

    def _embed_theses(self, date: int, theses: List[Thesis]) -> List[np.ndarray]:
        contents = [thesis.content for thesis in theses]
        embeddings = self.embedding_manager.embed(contents, partition=str(date))
        for thesis, embedding in zip(theses, embeddings):
            thesis.embedding = embedding
        return embeddings

    def _assign_thesis_scores(
        self,
        theses: List[Thesis],
        responsibilities: Optional[np.ndarray],
        predicted_scores: Optional[np.ndarray],
    ) -> None:
        if responsibilities is None or predicted_scores is None or predicted_scores.size == 0:
            for thesis in theses:
                thesis.score = 0.0
            return
        for thesis, omega in zip(theses, responsibilities):
            thesis.score = float(np.dot(omega, predicted_scores))

    def _aggregate_signals(self, date: int, theses: List[Thesis]) -> pd.DataFrame:
        bucket: Dict[str, Dict[str, List[float]]] = {}
        for thesis in theses:
            store = bucket.setdefault(thesis.stock, {"bullish": [], "bearish": []})
            key = "bullish" if thesis.polarity else "bearish"
            store[key].append(thesis.score)

        rows = []
        for stock, signals in bucket.items():
            bull = np.mean(signals["bullish"]) if signals["bullish"] else 0.0
            bear = np.mean(signals["bearish"]) if signals["bearish"] else 0.0
            rows.append({"Date": date, "Stock": stock, "signal": bull - bear})

        df = pd.DataFrame(rows)
        pool_frame = self.data_loader.get_pool_frame(date)
        if pool_frame.empty:
            return df
        merged = pool_frame.merge(df, on="Stock", how="left")
        merged["signal"] = merged["signal"].fillna(0.0)
        merged["Date"] = date
        return merged[["Date", "Stock", "signal"]]

    def _forward_fill_signals(self, end_date: Optional[int] = None) -> pd.DataFrame:
        final_frames: List[pd.DataFrame] = []
        last_signals = pd.DataFrame(columns=["Stock", "signal"])

        trading_dates = self.data_loader.trading_dates
        if end_date is not None:
            trading_dates = [d for d in trading_dates if d <= end_date]

        for date in trading_dates:
            pool_frame = self.data_loader.get_pool_frame(date)
            if pool_frame.empty:
                continue

            if date in self.pattern_signal_map:
                day_frame = self.pattern_signal_map[date].copy()
                last_signals = day_frame[["Stock", "signal"]]
            else:
                day_frame = pool_frame.merge(last_signals, on="Stock", how="left")
                day_frame["signal"] = day_frame["signal"].fillna(0.0)
                day_frame.insert(0, "Date", date)

            final_frames.append(day_frame[["Date", "Stock", "signal"]])

        if not final_frames:
            return pd.DataFrame(columns=["Date", "Stock", "signal"])
        return pd.concat(final_frames, ignore_index=True)

    def _log_cluster_quality(self, date: int, embeddings: np.ndarray, labels: np.ndarray) -> None:
        n_samples = embeddings.shape[0]
        unique_labels = np.unique(labels)
        if n_samples < 2 or unique_labels.size < 2:
            self._log_metrics(
                f"[CLUSTER] Date={date} K={self.pattern_state.n_components} insufficient samples for silhouette (n={n_samples})"
            )
            return
        try:
            score = silhouette_score(embeddings, labels)
        except Exception as exc:
            logger.warning("Failed to compute silhouette score on %s: %s", date, exc)
            score = float("nan")
        self._log_metrics(
            f"[CLUSTER] Date={date} K={self.pattern_state.n_components} silhouette={score:.4f} samples={n_samples}"
        )

    def _save_cluster_plot(self, date: int, embeddings: np.ndarray, labels: np.ndarray) -> None:
        n_samples = embeddings.shape[0]
        if n_samples < 5:
            return
        perplexity = min(30, max(5, n_samples // 3))
        perplexity = min(perplexity, n_samples - 1)
        if perplexity < 2:
            return
        try:
            tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=42)
            coords = tsne.fit_transform(embeddings)
        except Exception as exc:
            logger.warning("TSNE failed on %s: %s", date, exc)
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.75, label=f"C{label}")
        ax.set_title(f"{self.pool_name} clusters @ {date}")
        ax.set_xlabel("TSNE-1")
        ax.set_ylabel("TSNE-2")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        output_file = self.cluster_plot_dir / f"{date}_clusters_{self.pipeline_tag}.pdf"
        fig.savefig(output_file, format="pdf")
        plt.close(fig)
        self._log_metrics(f"[CLUSTER_PLOT] Date={date} saved TSNE visualization to {output_file}")

    def _log_cluster_migration(self, date: int, alignment: Optional[List[Tuple[int, int, float]]]) -> None:
        model = self.pattern_state.model
        if model is None or getattr(model, "means_", None) is None:
            return
        if not alignment:
            self._log_metrics(f"[CLUSTER_SHIFT] Date={date} initial clustering – no alignment available.")
            return
        new_means = model.means_
        pairwise = np.linalg.norm(new_means[:, None, :] - new_means[None, :, :], axis=2)
        summaries = []
        for new_idx, prev_idx, dist in alignment:
            others = np.delete(pairwise[new_idx], new_idx)
            avg_other = float(np.mean(others)) if others.size else float("nan")
            ratio = dist / avg_other if avg_other and not np.isnan(avg_other) and avg_other != 0 else float("nan")
            summaries.append(f"C{new_idx}←PrevC{prev_idx}: move={dist:.4f}, avg_gap={avg_other:.4f}, ratio={ratio:.4f}")
        self._log_metrics(f"[CLUSTER_SHIFT] Date={date} " + " | ".join(summaries))

    def _compute_signal_quality(self, date: int, day_signals: pd.DataFrame) -> None:
        if not self.selected_dates or self.selected_dates[-1] != date:
            self.selected_dates.append(date)
        returns = self.data_loader.stock_returns
        day_returns = returns[returns["Date"] == date][["Stock", "ret"]]
        merged = day_signals.merge(day_returns, on="Stock", how="inner")
        if merged.shape[0] < 2:
            self._log_metrics(f"[SIGNAL] Date={date} insufficient overlap for IC/RankIC (n={merged.shape[0]})")
            return
        ic = merged["signal"].corr(merged["ret"])
        rank_ic = merged["signal"].rank().corr(merged["ret"].rank())
        if not np.isnan(ic):
            self.ic_history.append(ic)
        if not np.isnan(rank_ic):
            self.rank_ic_history.append(rank_ic)

        ic_mean = np.nanmean(self.ic_history) if self.ic_history else float("nan")
        ic_std = np.nanstd(self.ic_history, ddof=1) if len(self.ic_history) > 1 else float("nan")
        ic_ir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) and ic_std != 0 else float("nan")
        rank_ic_mean = np.nanmean(self.rank_ic_history) if self.rank_ic_history else float("nan")
        rank_ic_std = np.nanstd(self.rank_ic_history, ddof=1) if len(self.rank_ic_history) > 1 else float("nan")
        rank_ic_ir = (
            rank_ic_mean / rank_ic_std if rank_ic_std and not np.isnan(rank_ic_std) and rank_ic_std != 0 else float("nan")
        )

        self._log_metrics(
            "[SIGNAL] Date={date} IC={ic:.4f} RankIC={rank_ic:.4f} | CumIC={cum_ic:.4f} ICIR={icir:.4f} "
            "CumRankIC={cum_rank:.4f} RankICIR={rank_ir:.4f}".format(
                date=date,
                ic=ic if not np.isnan(ic) else float("nan"),
                rank_ic=rank_ic if not np.isnan(rank_ic) else float("nan"),
                cum_ic=ic_mean if not np.isnan(ic_mean) else float("nan"),
                icir=ic_ir if not np.isnan(ic_ir) else float("nan"),
                cum_rank=rank_ic_mean if not np.isnan(rank_ic_mean) else float("nan"),
                rank_ir=rank_ic_ir if not np.isnan(rank_ic_ir) else float("nan"),
            )
        )

    def _log_metrics(self, message: str) -> None:
        logger.info(message)
        self.metrics_logger.info(message)
