"""
Configuration file for multi-agent stock selection framework
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class LLMConfig:
    model_name: str = "qwen3-max"
    max_tokens: int = 80000
    temperature: float = 0.0
    stop_words: List[str] = None
    max_retries: int = 10
    base_delay: float = 2.0
    
    def __post_init__(self):
        if self.stop_words is None:
            self.stop_words = []

@dataclass  
class ScreeningConfig:
    selection_ratio: float = 0.2  
    rebalance_frequency: int = 5  
    min_stock_count: int = 10     
    max_stock_count: int = 100    
    
@dataclass
class AnalysisConfig:
    lookback_window: int = 30     
    news_relevance_threshold: float = 0.7 
    max_news_per_stock: int = 10 
    analysis_cache_days: int = 1  
    
@dataclass
class DecisionConfig:
    fundamental_weight: float = 0.4 
    technical_weight: float = 0.25
    news_weight: float = 0.2
    industry_weight: float = 0.0   
    macro_weight: float = 0.15
    
    max_position: float = 1.0    
    single_stock_limit: float = 0.1 
    min_score_threshold: float = 60   
    
@dataclass
class ReflectionConfig:
    macro_reflection_window: int = 60      
    trading_reflection_window: int = 20    
    micro_reflection_window: int = 10      
    reflection_frequency: int = 5          
    
@dataclass
class DataConfig:
    data_root: str = "/PATH/TO/data"
    res_root: str = "/PATH/TO/res"
    all_features_file: str = "all_features.parq"
    stock_labels_file: str = "stock_labels.parq"
    income_file: str = "income.parq"
    balance_sheet_file: str = "balance_sheet.parq"
    index_classification_file: str = "index_classification.parq"
    base_data_file: str = "base_data.parq"
    base_data_with_macd_file: str = "base_data_with_macd.parq"
    news_dir: str = "news"
    index_news_dir: str = "news/index"
    features_file: str = "all_features.parq"
    index_map = {"csi_300": "000300.SH", "csi_500": "000905.SH", "csi_1000": "000852.SH"}
    
    @property
    def all_features_path(self) -> str:
        return os.path.join(self.data_root, self.all_features_file)
    
    @property
    def stock_labels_path(self) -> str:
        return os.path.join(self.data_root, self.stock_labels_file)
    
    @property
    def industry_returns_path(self) -> str:
        return os.path.join(self.data_root, f"industry_ret")
    
    @property
    def analysis_res_path(self) -> str:
        return os.path.join(self.res_root, "analysis_res")
    
    @property
    def index_returns_path(self) -> str:
        return os.path.join(self.data_root, f"index_ret")
    
    @property
    def index_pool_path(self) -> str:
        return os.path.join(self.data_root, f"{DEFAULT_CONFIG.benchmark_index}.parq")

    @property
    def fundamental_data_path(self) -> str:
        return os.path.join(self.data_root, "fundamental_data.parq")
    
    @property
    def features_path(self) -> str:
        return os.path.join(self.data_root, self.features_file)
    
    @property
    def industry_fundamental_path(self) -> str:
        return os.path.join(self.data_root, f"industry_fundamental")
    
    @property
    def index_fundamental_path(self) -> str:
        return os.path.join(self.data_root, f"index_fundamental")
    
    @property
    def benchmark_index_with_macd_path(self) -> str:
        return os.path.join(self.data_root, f"{DEFAULT_CONFIG.benchmark_index}_base_data_with_macd.parq")

    @property
    def income_path(self) -> str:
        return os.path.join(self.data_root, self.income_file)
        
    @property
    def balance_path(self) -> str:
        return os.path.join(self.data_root, self.balance_sheet_file)
        
    @property
    def industry_path(self) -> str:
        return os.path.join(self.data_root, self.index_classification_file)
        
    @property 
    def base_data_path(self) -> str:
        return os.path.join(self.data_root, self.base_data_file)

    @property
    def base_data_with_macd_path(self) -> str:
        return os.path.join(self.data_root, self.base_data_with_macd_file)
        
    @property
    def news_path(self) -> str:
        return os.path.join(self.data_root, self.news_dir)
    
    @property
    def index_news_path(self) -> str:
        return os.path.join(self.data_root, self.index_news_dir)

@dataclass
class GlobalConfig:
    """全局配置"""
    llm: LLMConfig = None
    screening: ScreeningConfig = None
    analysis: AnalysisConfig = None
    decision: DecisionConfig = None
    reflection: ReflectionConfig = None
    data: DataConfig = None
    
    # 实验配置
    start_date: int = 20220101
    end_date: int = 20241231
    benchmark_index: str = "csi_300"  
    
    def __post_init__(self):
        if self.llm is None:
            self.llm = LLMConfig()
        if self.screening is None:
            self.screening = ScreeningConfig()
        if self.analysis is None:
            self.analysis = AnalysisConfig()
        if self.decision is None:
            self.decision = DecisionConfig()
        if self.reflection is None:
            self.reflection = ReflectionConfig()
        if self.data is None:
            self.data = DataConfig()

# 默认配置实例
DEFAULT_CONFIG = GlobalConfig()
