"""
股票数据加载器
Stock data loader for comprehensive stock information
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import os

from multi_agent_stock_selection.config.config import GlobalConfig
from ..utils.date_utils import get_begin_date

logger = logging.getLogger(__name__)

class StockDataLoader:
    """
    综合股票数据加载器
    """
    
    def __init__(self, config: GlobalConfig, used_index_classification: str = "sw_l2", index: str = "csi_300",
    stock_labels: pd.DataFrame = None, index_classification: pd.DataFrame = None):
        """
        初始化数据加载器
        
        Args:
            data_config: 数据配置对象
        """
        self.config = config
        self.stock_labels = stock_labels
        self.index_classification = index_classification
        self.base_data = pd.read_parquet(self.config.data.base_data_path, columns=["Stock", "Date", "Name", "MV", "FREE_MV"])
        self.base_data_with_macd = pd.read_parquet(self.config.data.base_data_with_macd_path)
        self.benchmark_index_with_macd = pd.read_parquet(self.config.data.benchmark_index_with_macd_path)
        self.used_index_classification = used_index_classification
        self.index = config.benchmark_index
        self._cache = {}
        
        # 预定义列列表，避免重复定义
        self._used_keys_d = ["Open", "High", "Low", "Close", "Value","MV_5_D", "MV_10_D", "MV_20_D", "MV_60_D", 
                            "MV_120_D", "MV_250_D", "DIF_D", "DEA_D", "MACD_D",
                            "BOLL_UPPER_D", "BOLL_MID_D", "BOLL_LOWER_D",
                            "RSI_D"]
        self._used_keys_w = ["Open_W", "High_W", "Low_W", "Close_W","Value_W", "MV_5_W", "MV_10_W", "MV_20_W", "MV_60_W",
                            "MV_120_W", "DIF_W", "DEA_W", "MACD_W",
                            "BOLL_UPPER_W", "BOLL_MID_W", "BOLL_LOWER_W",
                            "RSI_W"]
        self._used_keys_m = ["Open_M", "High_M", "Low_M", "Close_M","Value_M", "MV_5_M", "MV_10_M", "MV_20_M", "MV_60_M",
                           "DIF_M", "DEA_M", "MACD_M",
                            "BOLL_UPPER_M", "BOLL_MID_M", "BOLL_LOWER_M",
                            "RSI_M"]
        
        # 预计算需要的所有列，只需要计算一次
        self._all_daily_cols = ["Stock", "Date"] + self._used_keys_d
        self._all_weekly_cols = ["Stock", "Date"] + self._used_keys_w
        self._all_monthly_cols = ["Stock", "Date"] + self._used_keys_m
    
    def load_stock_basic_info(self, stock_code: str, date: int) -> dict[str, Any]:
        """
        加载股票基础信息
        
        Args:
            stock_code: 股票代码
            date: 日期
            
        Returns:
            股票基础信息字典
        """
        try:
            # 加载基础行情数据
            current_stock_basic_data = self.base_data[(self.base_data['Stock'] == stock_code) & (self.base_data['Date'] <= date)][["Stock", "Date", "Name", "MV", "FREE_MV"]].copy(deep=False)
            return {"stock_name": current_stock_basic_data["Name"].iloc[-1], "code": stock_code, "mv": current_stock_basic_data["MV"].iloc[-1], "free_mv": current_stock_basic_data["FREE_MV"].iloc[-1]}
            
        except Exception as e:
            logger.error(f"加载股票基础信息失败: {stock_code}, {e}")
            return {}
    
    def load_stock_returns(
        self, 
        stock_code: str, 
        start_date: int, 
        end_date: int
    ) -> pd.DataFrame:
        """
        加载股票收益率数据.
        使用stock_labels的1_15_labelB字段，避免未来信息泄露
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            收益率数据DataFrame
        """
        try:
            
            # 筛选股票和时间范围，向前平移一天避免未来信息
            stock_data = self.stock_labels[
                (self.stock_labels['Stock'] == stock_code) &
                (self.stock_labels['Date'] >= get_begin_date(start_date, 0, 0, 1)) &
                (self.stock_labels['Date'] <= get_begin_date(end_date, 0, 0, 1))  # 向前平移避免未来信息
            ].sort_values('Date').copy(deep=False)
            
            if stock_data.empty:
                return pd.DataFrame()
            
            # # 使用1_15_labelB作为日收益率
            # stock_data['ret'] = stock_data['1_15_labelB']
            
            
            return stock_data[['Date', '1_15_labelB']].rename(columns={'1_15_labelB': 'ret'})
            
        except Exception as e:
            logger.error(f"加载股票收益率失败: {stock_code}, {e}")
            return pd.DataFrame()
    
    def load_industry_returns(self, stock_code: str, start_date: int, end_date: int) -> Tuple[str, pd.DataFrame]:
        """
        加载行业收益率数据.
        stock_code:股票代码
        start_date: 开始日期
        end_date: 结束日期
        """
        # 优化：一次性获取行业代码和名称，避免重复过滤
        industry_info = self.index_classification[
            (self.index_classification['index_classification'] == self.used_index_classification) & 
            (self.index_classification['Stock'] == stock_code) &
            (self.index_classification["Date"] <= end_date)
        ][['industry_code', 'industry_name']].iloc[-1]
        
        industry_code = industry_info['industry_code']
        industry_name = industry_info['industry_name']
        
        # 预计算文件路径，避免重复构建
        industry_file_path = os.path.join(self.config.data.industry_returns_path, f"{industry_code}_ret.parq")
        
        if os.path.exists(industry_file_path):
            industry_returns = pd.read_parquet(industry_file_path)
        else:
            sub_index_classification = self.index_classification[
                (self.index_classification['index_classification'] == self.used_index_classification) & 
                (self.index_classification['industry_code'] == industry_code)
            ]
            industry_returns = self.stock_labels.merge(
                sub_index_classification[["Stock", "Date"]], 
                on=["Stock", "Date"], 
                how="inner"
            ).groupby("Date", as_index=False)["1_15_labelB"].mean().rename(columns={"1_15_labelB": "ret"})
            industry_returns.to_parquet(industry_file_path)
        
        # 优化：移除不必要的copy操作，直接使用布尔索引
        start_date_adj = get_begin_date(start_date, 0, 0, 1)
        end_date_adj = get_begin_date(end_date, 0, 0, 1)
        sub_ret = industry_returns[
            (industry_returns['Date'] >= start_date_adj) & 
            (industry_returns['Date'] <= end_date_adj)
        ]
        
        return industry_name, sub_ret[["Date", "ret"]]
    
    def load_index_returns(self, start_date: int, end_date: int) -> pd.DataFrame:
        """
        加载指数收益率数据.
        start_date: 开始日期
        end_date: 结束日期
        """
        if os.path.exists(os.path.join(self.config.data.index_returns_path, f"{self.index}_ret.parq")):
            index_returns = pd.read_parquet(os.path.join(self.config.data.index_returns_path, f"{self.index}_ret.parq"))
        else:
            stock_pool = pd.read_parquet(os.path.join(self.config.data.data_root, f"{self.index}.parq"))
            index_returns = self.stock_labels.merge(stock_pool[["Stock", "Date"]], on=["Stock", "Date"], how="inner").groupby("Date", as_index=False)["1_15_labelB"].mean().rename(columns={"1_15_labelB": "ret"})
            index_returns.to_parquet(os.path.join(self.config.data.index_returns_path, f"{self.index}_ret.parq"))
        sub_ret = index_returns[(index_returns['Date'] >= get_begin_date(start_date, 0, 0, 1)) & (index_returns['Date'] <= get_begin_date(end_date, 0, 0, 1))].copy(deep=False)
        # sub_ret["cumulative_return"] = sub_ret["ret"].cumsum()
        return sub_ret[["Date", "ret"]]
    
    def load_benchmark_index_with_macd(self, end_date: int, look_back_period: int = 3) -> pd.DataFrame:
        """
        加载基准指数基础数据和MACD数据.
        start_date: 开始日期
        end_date: 结束日期
        """
        # 优化：一次性获取和排序数据
        sub_stock_frame = self.benchmark_index_with_macd[self.benchmark_index_with_macd['Date'] <= end_date].sort_values('Date')
        
        # 计算最大需要的行数
        n_monthly = (look_back_period - 1) * 20 + 1
        total_rows = len(sub_stock_frame)
        
        if total_rows == 0:
            empty_daily = pd.DataFrame(columns=self._all_daily_cols)
            empty_weekly = pd.DataFrame(columns=self._all_weekly_cols) 
            empty_monthly = pd.DataFrame(columns=self._all_monthly_cols)
            return empty_daily, empty_weekly, empty_monthly
        
        # 一次性获取所有需要的行
        start_idx = max(0, total_rows - n_monthly)
        required_data = sub_stock_frame.iloc[start_idx:]
        
        # 日线数据
        daily_return_frame = required_data[self._all_daily_cols].tail(look_back_period)
        
        # 周线数据 - 优化采样
        n_weekly = (look_back_period - 1) * 5 + 1
        weekly_candidate = required_data[self._all_weekly_cols].tail(n_weekly)
        weekly_return_frame = weekly_candidate.iloc[::-5].iloc[::-1]
        
        # 月线数据 - 优化采样
        monthly_candidate = required_data[self._all_monthly_cols]
        monthly_return_frame = monthly_candidate.iloc[::-20].iloc[::-1]
            
        return daily_return_frame, weekly_return_frame, monthly_return_frame
    

    def load_base_data_with_macd(self, stock_code: str, date: int, look_back_period: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载基础数据和MACD数据.
        stock_code: 股票代码
        date: 日期
        look_back_period: 回看周期 包括日线 周线 月线的回看周期
        """
        # 优化1: 使用query方法，比布尔索引更快
        # 优化2: 直接选择需要的最大列数，避免多次列选择
        # 计算最大需要的行数
        n_monthly = (look_back_period - 1) * 20 + 1
        
        # 一次性获取所有需要的数据，避免重复过滤和排序
        stock_mask = self.base_data_with_macd['Stock'] == stock_code
        date_mask = self.base_data_with_macd['Date'] <= date
        sub_stock_frame = self.base_data_with_macd.loc[stock_mask & date_mask].sort_values('Date')
        
        # 优化3: 直接使用iloc进行快速切片，避免copy操作
        total_rows = len(sub_stock_frame)
        if total_rows == 0:
            # 返回空的DataFrame，保持原始结构
            empty_daily = pd.DataFrame(columns=self._all_daily_cols)
            empty_weekly = pd.DataFrame(columns=self._all_weekly_cols)
            empty_monthly = pd.DataFrame(columns=self._all_monthly_cols)
            return empty_daily, empty_weekly, empty_monthly
        
        # 优化4: 一次性获取所有需要的行，然后进行切片
        # 确保不超出数据范围
        start_idx = max(0, total_rows - n_monthly)
        required_data = sub_stock_frame.iloc[start_idx:]
        
        # 日线数据 - 取最后look_back_period行
        daily_return_frame = required_data[self._all_daily_cols].tail(look_back_period)
        
        # 周线数据 - 优化采样逻辑
        n_weekly = (look_back_period - 1) * 5 + 1
        weekly_candidate = required_data[self._all_weekly_cols].tail(n_weekly)
        weekly_return_frame = weekly_candidate.iloc[::-5].iloc[::-1]
        
        # 月线数据 - 优化采样逻辑
        monthly_candidate = required_data[self._all_monthly_cols]
        monthly_return_frame = monthly_candidate.iloc[::-20].iloc[::-1]
            
        return daily_return_frame, weekly_return_frame, monthly_return_frame

