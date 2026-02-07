"""
Command-line entry point for stage-two pattern clustering.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from types import MethodType
from typing import Optional

import numpy as np
import pandas as pd

from multi_agent_stock_selection.config.config import GlobalConfig
from multi_agent_stock_selection.patterns.pipeline import PatternClusteringPipeline


class PatternScoreTracker:
    """Tracks per-date pattern scores with stable IDs across cross-period alignment."""

    def __init__(self) -> None:
        self.stable_map: dict[int, int] = {}
        self.next_id: int = 0
        self.records: list[dict[str, float | int]] = []

    def record(self, date: int, alignment, p_scores: np.ndarray) -> None:
        if p_scores is None:
            return
        scores = np.asarray(p_scores, dtype=float)
        if scores.size == 0:
            return

        if not self.stable_map:
            self.stable_map = {idx: idx for idx in range(scores.shape[0])}
            self.next_id = scores.shape[0]

        new_map: dict[int, int] = {}
        if alignment:
            for new_idx, prev_idx, _ in alignment:
                stable_id = self.stable_map.get(prev_idx)
                if stable_id is None:
                    stable_id = self.next_id
                    self.next_id += 1
                new_map[new_idx] = stable_id

        for idx in range(scores.shape[0]):
            if idx not in new_map:
                stable_id = self.stable_map.get(idx)
                if stable_id is None:
                    stable_id = self.next_id
                    self.next_id += 1
                new_map[idx] = stable_id

        self.stable_map = new_map
        for idx, score in enumerate(scores):
            self.records.append({"Date": int(date), "pattern_id": int(new_map[idx]), "score": float(score)})

    def to_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["Date", "pattern_id", "score"])
        return pd.DataFrame(self.records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thought-pattern based multi-agent stock selection (stage two).")
    parser.add_argument("--pool-name", type=str, default="csi_500", help="Stock pool to process.", choices=["sse_50", "csi_300", "csi_500"])
    parser.add_argument("--start-date", type=int, default=20250924, help="Backtest start date (YYYYMMDD).")
    parser.add_argument("--end-date", type=int, default=20251231, help="Backtest end date (YYYYMMDD).")
    parser.add_argument("--days-per-pattern", type=int, default=5, help="Number of trading days between pattern updates.")
    parser.add_argument("--num-patterns", type=int, default=20, help="Number of GMM components (patterns).")
    parser.add_argument("--return-column", type=str, default="5_15_labelB", help="Return column to evaluate thesis PnL.")
    parser.add_argument("--decay-lambda", type=float, default=0.5, help="Exponential decay factor for long-term P-score.")
    parser.add_argument("--beta", type=float, default=0.0, help="Distance decay coefficient for predicted pattern performance.")
    parser.add_argument("--embedding-api-key", type=str, default="embedding-api-key", help="OpenRouter API key (fallback: OPENROUTER_API_KEY).")
    parser.add_argument("--embedding-model", type=str, default="qwen/qwen3-embedding-8b", help="Embedding model name.")
    parser.add_argument("--embedding-batch-size", type=int, default=256, help="Batch size for embedding requests.")
    parser.add_argument("--embedding-cache-dir", "--embedding-cache", dest="embedding_cache_dir", type=str, default=None, help="Directory to store per-date embedding caches. Defaults to res/pattern_cache/<pool>.")
    parser.add_argument("--disable-embedding-cache", action="store_true", help="Disable reading/writing local embedding caches.")
    parser.add_argument("--distance-metric", type=str, default="euclidean", choices=("euclidean", "manhattan"), help="Distance metric for cross-period Hungarian matching.")
    parser.add_argument("--output-path", type=str, default=None, help="Output parquet path. Defaults to res/pattern_signals/<pool>.parq")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser


def resolve_api_key(args_api_key: Optional[str]) -> str:
    api_key = args_api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OpenRouter API key is required. Pass --embedding-api-key or set OPENROUTER_API_KEY.")
    return api_key


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    api_key = resolve_api_key(args.embedding_api_key)

    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(
            "res/pattern_signals/"
            f"{args.pool_name}_signal_{args.num_patterns}_{args.decay_lambda}_{args.beta}_signals_2025_Q4.parq"
        )
    )

    config = GlobalConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_index=args.pool_name,
    )

    cache_dir = None if args.disable_embedding_cache else (args.embedding_cache_dir or f"res/pattern_cache/{args.pool_name}")

    pipeline = PatternClusteringPipeline(
        config=config,
        pool_name=args.pool_name,
        start_date=args.start_date,
        end_date=args.end_date,
        days_per_pattern=args.days_per_pattern,
        num_patterns=args.num_patterns,
        return_column=args.return_column,
        decay_lambda=args.decay_lambda,
        beta=args.beta,
        distance_metric=args.distance_metric,
        embedding_api_key=api_key,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        embedding_cache_dir=cache_dir,
        output_path=output_path,
    )

    score_tracker = PatternScoreTracker()
    orig_log_shift = pipeline._log_cluster_migration.__func__

    def _log_cluster_migration_with_tracking(self, date: int, alignment):
        score_tracker.record(date, alignment, self.pattern_state.p_scores)
        return orig_log_shift(self, date, alignment)

    pipeline._log_cluster_migration = MethodType(_log_cluster_migration_with_tracking, pipeline)

    final_df = pipeline.run()
    if final_df.empty:
        logging.warning("No signals generated.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(final_df, output_path)
    score_df = score_tracker.to_frame()
    if not score_df.empty:
        score_dir = Path("pattern_state")
        score_dir.mkdir(parents=True, exist_ok=True)
        score_path = score_dir / f"{pipeline.pipeline_tag}_p_scores.parq"
        score_df.to_parquet(score_path, index=False)
        logging.info("Saved pattern score traces to %s", score_path)
    logging.info("Pipeline finished with %d dates from %s to %s.", len(final_df["Date"].unique()), args.start_date, args.end_date)


if __name__ == "__main__":
    main()
