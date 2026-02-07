import argparse
import os
from tqdm import tqdm
from multi_agent_stock_selection import *
from typing import Any
from openai import OpenAI
from multi_agent_stock_selection.layers import AnalysisLayer
from multi_agent_stock_selection.utils.llm import OpenAIModel
from multi_agent_stock_selection.config.config import GlobalConfig
import pandas as pd


POOL_MAPPING = {
    "csi_300": "000300.SH",
    "csi_500": "000905.SH",
    "csi_1000": "000852.SH",
}

def train(args: Any):
    stock_pool = POOL_MAPPING.get(args.stock_pool, args.stock_pool)
    config = GlobalConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_index=args.stock_pool,
    )
    openai = OpenAI(api_key=args.api_key, base_url="model_url")
    model = OpenAIModel(model_name=args.model_name, stop_words="", max_new_tokens=80000)
    
    analysis_layer = AnalysisLayer(config=config, open_ai=openai, open_ai_model=model)
    base_data = pd.read_parquet(f"data/{args.stock_pool}.parq", columns=["Date", "Stock"], filters=[("Date", ">=", args.start_date), ("Date", "<=", args.end_date)])
    dates = sorted(base_data["Date"].unique().tolist())
    sub_dates = dates[::args.days_per_pattern]
    for date in tqdm(sub_dates, desc="Generating patterns on each date"):
        stocks_on_date = base_data[base_data["Date"] == date]["Stock"].tolist()
        analysis_layer.analyze(stock_codes=stocks_on_date, date=date, parrell=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent portfolio management through pattern.")
    parser.add_argument("--stock_pool", type=str, default="csi_500", help="Path to the configuration file.")
    parser.add_argument("--start_date", type=int, default=20251009)
    parser.add_argument("--end_date", type=int, default=20251231)
    parser.add_argument("--output_dir", type=str, default="results/pattern_training")
    parser.add_argument("--num_patterns", type=int, default=10)
    parser.add_argument("--days_per_pattern", type=int, default=5)
    parser.add_argument("--pattern_decay", type=float, default=0.9)
    parser.add_argument("--model_name", type=str, default="deepseek-v3.1")
    parser.add_argument("--api_key", type=str, default="api key")
    args = parser.parse_args()
    train(args)