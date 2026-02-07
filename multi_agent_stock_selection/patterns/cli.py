"""
Command-line entry point for stage-two pattern clustering.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from multi_agent_stock_selection.config.config import GlobalConfig
from multi_agent_stock_selection.patterns.pipeline import PatternClusteringPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thought-pattern based multi-agent stock selection (stage two).")
    parser.add_argument("--pool-name", type=str, default="csi_300", help="Stock pool to process.")
    parser.add_argument("--start-date", type=int, default=20231001, help="Backtest start date (YYYYMMDD).")
    parser.add_argument("--end-date", type=int, default=20250930, help="Backtest end date (YYYYMMDD).")
    parser.add_argument("--days-per-pattern", type=int, default=5, help="Number of trading days between pattern updates.")
    parser.add_argument("--num-patterns", type=int, default=15, help="Number of GMM components (patterns).")
    parser.add_argument("--return-column", type=str, default="5_15_labelB", help="Return column to evaluate thesis PnL.")
    parser.add_argument("--decay-lambda", type=float, default=0.9, help="Exponential decay factor for long-term P-score.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Blend weight between P-score and N-score.")
    parser.add_argument("--embedding-api-key", type=str, default=None, help="OpenRouter API key (fallback: OPENROUTER_API_KEY).")
    parser.add_argument("--embedding-model", type=str, default="qwen/qwen3-embedding-8b", help="Embedding model name.")
    parser.add_argument("--embedding-batch-size", type=int, default=128, help="Batch size for embedding requests.")
    parser.add_argument("--embedding-cache", type=str, default="res/pattern_cache/embeddings.pkl", help="Optional pickle cache for embeddings.")
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

    output_path = Path(args.output_path) if args.output_path else Path(f"res/pattern_signals/{args.pool_name}_signals.parq")

    config = GlobalConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_index=args.pool_name,
    )

    pipeline = PatternClusteringPipeline(
        config=config,
        pool_name=args.pool_name,
        start_date=args.start_date,
        end_date=args.end_date,
        days_per_pattern=args.days_per_pattern,
        num_patterns=args.num_patterns,
        return_column=args.return_column,
        decay_lambda=args.decay_lambda,
        gamma=args.gamma,
        embedding_api_key=api_key,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        embedding_cache_path=args.embedding_cache or None,
        output_path=output_path,
    )

    final_df = pipeline.run()
    if final_df.empty:
        logging.warning("No signals generated.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(final_df, output_path)
    logging.info("Pipeline finished with %d dates from %s to %s.", len(final_df["Date"].unique()), args.start_date, args.end_date)


if __name__ == "__main__":
    main()
