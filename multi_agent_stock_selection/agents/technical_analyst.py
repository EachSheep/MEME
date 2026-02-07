"""
技术分析师
Technical analyst for stock technical analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

from multi_agent_stock_selection.utils.date_utils import get_begin_date

from .base_analyst import BaseAnalyst

logger = logging.getLogger(__name__)
from openai import OpenAI
from ..utils.llm import OpenAIModel
from ..config.config import GlobalConfig
from ..data_loaders.stock_data_loader import StockDataLoader
from ..utils.prompt import PROMPT_TECHNICALS, PROMPT_INSTRUCTIONS

class TechnicalAnalyst(BaseAnalyst):
    def __init__(self, llm_client: OpenAI, model: OpenAIModel, config: GlobalConfig=None, stock_labels: pd.DataFrame = None, index_classification: pd.DataFrame = None):
        super().__init__(llm_client=llm_client, model=model, config=config, analyst_name=self.__class__.__name__)
        self.stock_loader = StockDataLoader(config, index=config.benchmark_index, stock_labels=stock_labels, index_classification=index_classification)
        
    
    def _generate_analysis_prompt_from_dataframe(self, data: pd.DataFrame) -> str:
        return data.to_string(index=False)
    
    def get_analysis_prompt(self, stock_code: str, analysis_date: int, **kwargs) -> str:
        basic_info = self.stock_loader.load_stock_basic_info(stock_code, analysis_date)
        stock_name = basic_info["stock_name"]
        
        # Load data once
        daily_technical_data, weekly_technical_data, monthly_technical_data = self.stock_loader.load_base_data_with_macd(stock_code, analysis_date)
        stock_return = self.stock_loader.load_stock_returns(stock_code, get_begin_date(analysis_date, 1, 0, 3), analysis_date)
        industry_name, industry_return = self.stock_loader.load_industry_returns(stock_code, get_begin_date(analysis_date, 1, 0, 3), analysis_date)
        index_return = self.stock_loader.load_index_returns(get_begin_date(analysis_date, 1, 0, 3), analysis_date)
        # Pre-calculate returns using correct windowing logic
        yearly_ret_len = min(stock_return.shape[0], 250)
        half_year_ret_len = min(stock_return.shape[0], 121)
        one_month_ret_len = min(stock_return.shape[0], 21)
        one_week_ret_len = min(stock_return.shape[0], 5)
        
        yearly_stock_ret = stock_return["ret"].iloc[-yearly_ret_len:].cumsum().iloc[-1]
        yearly_industry_ret = industry_return["ret"].iloc[-yearly_ret_len:].cumsum().iloc[-1]
        yearly_index_ret = index_return["ret"].iloc[-yearly_ret_len:].cumsum().iloc[-1]

        half_year_stock_ret = stock_return["ret"].iloc[-half_year_ret_len:].cumsum().iloc[-1]
        half_year_industry_ret = industry_return["ret"].iloc[-half_year_ret_len:].cumsum().iloc[-1]
        half_year_index_ret = index_return["ret"].iloc[-half_year_ret_len:].cumsum().iloc[-1]

        one_month_stock_ret = stock_return["ret"].iloc[-one_month_ret_len:].cumsum().iloc[-1]
        one_month_industry_ret = industry_return["ret"].iloc[-one_month_ret_len:].cumsum().iloc[-1]
        one_month_index_ret = index_return["ret"].iloc[-one_month_ret_len:].cumsum().iloc[-1]

        one_week_stock_ret = stock_return["ret"].iloc[-one_week_ret_len:].cumsum().iloc[-1]
        one_week_industry_ret = industry_return["ret"].iloc[-one_week_ret_len:].cumsum().iloc[-1]
        one_week_index_ret = index_return["ret"].iloc[-one_week_ret_len:].cumsum().iloc[-1]
        
        # Pre-define column lists to avoid recreation
        daily_cols = ["Date", "Open", "High", "Low", "Close", "Value", "MV_5_D", "MV_10_D", "MV_20_D", 
                     "MV_60_D", "MV_120_D", "MV_250_D", "DIF_D", "DEA_D", "MACD_D", "BOLL_UPPER_D", 
                     "BOLL_MID_D", "BOLL_LOWER_D", "RSI_D"]
        weekly_cols = ["Date", "Open_W", "High_W", "Low_W", "Close_W", "Value_W", "MV_5_W", "MV_10_W", 
                      "MV_20_W", "MV_60_W", "MV_120_W", "DIF_W", "DEA_W", "MACD_W", "BOLL_UPPER_W", 
                      "BOLL_MID_W", "BOLL_LOWER_W", "RSI_W"]
        monthly_cols = ["Date", "Open_M", "High_M", "Low_M", "Close_M", "Value_M", "MV_5_M", "MV_10_M", 
                       "MV_20_M", "MV_60_M", "DIF_M", "DEA_M", "MACD_M", "BOLL_UPPER_M", "BOLL_MID_M", 
                       "BOLL_LOWER_M", "RSI_M"]
        
        # Convert DataFrames to strings once
        daily_data_str = self._generate_analysis_prompt_from_dataframe(daily_technical_data[daily_cols])
        weekly_data_str = self._generate_analysis_prompt_from_dataframe(weekly_technical_data[weekly_cols])
        monthly_data_str = self._generate_analysis_prompt_from_dataframe(monthly_technical_data[monthly_cols])
        
        # Build prompt using list for efficient string concatenation
        prompt_parts = [
            f"""
                     {PROMPT_TECHNICALS}\n"
        """,
            f"""
                      "对于股票 代码{stock_code} 名称{stock_name}, 其日线级别基础行情信息和技术面指标信息如下 你应分析最近5个交易日的高开低收 成交量 与均线 布林线 RSI的关系 以及这些技术指标所揭示出来的信息：\n\n"
                      "字段含义为 Open: 开盘价 High: 最高价 Low: 最低价 Close: 收盘价 Value: 成交量
                      MV_5_D: 5日均线 MV_10_D: 10日均线 MV_20_D: 20日均线 MV_60_D: 60日均线 MV_120_D: 120日均线 MV_250_D: 250日均线
                      DIF_D: 差离值 DEA_D: 讯号线 MACD_D: 平滑移动平均线 BOLL_UPPER_D: 布林线上轨 BOLL_MID_D: 布林线中轨 BOLL_LOWER_D: 布林线下轨
                      RSI_D: 相对强弱指数\n"
                      """,
            daily_data_str,
            f"""
                      "对于股票 代码{stock_code} 名称{stock_name}, 其周线级别基础行情信息和技术面指标信息如下 你应分析最近5个交易周的高开低收 成交量 与均线 布林线 RSI的关系 以及这些技术指标所揭示出来的信息：\n\n"
                      "字段含义为 Open_W: 开盘价 High_W: 最高价 Low_W: 最低价 Close_W: 收盘价 Value_W: 成交量
                      MV_5_W: 5日均线 MV_10_W: 10日均线 MV_20_W: 20日均线 MV_60_W: 60日均线 MV_120_W: 120日均线
                      DIF_W: 差离值 DEA_W: 讯号线 MACD_W: 平滑移动平均线 BOLL_UPPER_W: 布林线上轨 BOLL_MID_W: 布林线中轨 BOLL_LOWER_W: 布林线下轨
                      RSI_W: 相对强弱指数\n"
                      """,
            weekly_data_str,
            f"""
                      "对于股票 代码{stock_code} 名称{stock_name}, 其月线级别基础行情信息和技术面指标信息如下 你应分析最近5个交易月的高开低收 成交量 与均线 布林线 RSI的关系 以及这些技术指标所揭示出来的信息：\n\n"
                      "字段含义为 Open_M: 开盘价 High_M: 最高价 Low_M: 最低价 Close_M: 收盘价 Value_M: 成交量
                      MV_5_M: 5日均线 MV_10_M: 10日均线 MV_20_M: 20日均线 MV_60_M: 60日均线
                      DIF_M: 差离值 DEA_M: 讯号线 MACD_M: 平滑移动平均线 BOLL_UPPER_M: 布林线上轨 BOLL_MID_M: 布林线中轨 BOLL_LOWER_M: 布林线下轨
                      RSI_M: 相对强弱指数\n"
                      """,
            monthly_data_str,
            f"""
                       对于 股票 代码{stock_code} 名称{stock_name}, 近一年来的走势统计如下:
                       近一年来 {stock_code} 涨幅为 {yearly_stock_ret * 100:.2f}%, {industry_name} 行业涨幅为 {yearly_industry_ret * 100:.2f}%, {self.stock_loader.index} 指数涨幅为 {yearly_index_ret * 100:.2f}%
                       近半年以来 {stock_code} 涨幅为 {half_year_stock_ret * 100:.2f}%, {industry_name} 行业涨幅为 {half_year_industry_ret * 100:.2f}%, {self.stock_loader.index} 指数涨幅为 {half_year_index_ret * 100:.2f}%
                       近一个月以来 {stock_code} 涨幅为 {one_month_stock_ret * 100:.2f}%, {industry_name} 行业涨幅为 {one_month_industry_ret * 100:.2f}%, {self.stock_loader.index} 指数涨幅为 {one_month_index_ret * 100:.2f}%
                       近一周以来 {stock_code} 涨幅为 {one_week_stock_ret * 100:.2f}%, {industry_name} 行业涨幅为 {one_week_industry_ret * 100:.2f}%, {self.stock_loader.index} 指数涨幅为 {one_week_index_ret * 100:.2f}%\n
                       """,
            PROMPT_INSTRUCTIONS
        ]
        
        return ''.join(prompt_parts)


    
    
    # def get_analysis_prompt(self, stock_code: str, analysis_date: int, **kwargs) -> str:
    #     stock_basic_info = self.stock_loader.load_stock_basic_info(stock_code, analysis_date)  
    #     stock_name = stock_basic_info["stock_name"]                                                                                    
    #     technical_data = self.stock_loader.load_technical_data(stock_code, analysis_date)
    #     return f"""请分析股票{stock_name}（代码：{stock_code}）在{analysis_date} 的技术指标信息 
    #     判断这些技术指标是否可能影响到, 以及如何影响到 {stock_name}未来 在二级市场的未来走势. 
    #     """
