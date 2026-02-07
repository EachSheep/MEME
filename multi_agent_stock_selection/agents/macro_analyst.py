"""
宏观分析师
Macro analyst for benchmark index.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from .base_analyst import BaseAnalyst
from multi_agent_stock_selection.data_loaders.stock_data_loader import StockDataLoader
from multi_agent_stock_selection.config.config import GlobalConfig
from multi_agent_stock_selection.data_loaders.news_data_loader import NewsDataLoader
from multi_agent_stock_selection.data_loaders.financial_data_loader import FinancialDataLoader

logger = logging.getLogger(__name__)

class MacroAnalyst(BaseAnalyst):
    def __init__(self, llm_client: Any, model: str, config: GlobalConfig, stock_labels: pd.DataFrame = None, index_classification: pd.DataFrame = None):
        super().__init__(llm_client=llm_client, model=model, config=config, analyst_name= self.__class__.__name__)
        self.stock_loader = StockDataLoader(config, stock_labels=stock_labels, index_classification=index_classification)
        self.news_loader = NewsDataLoader(config, use_index=True)
        self.financial_loader = FinancialDataLoader(config, index_classification=index_classification)

    def _generate_analysis_prompt_from_dataframe(self, data: pd.DataFrame) -> str:
        return data.to_string(index=False)
    

    def generate_analysis_prompt_from_stock_fundamentals(self, stock_code: str, stock_name: str, 
                                                          fundamental_frame: pd.DataFrame,
                                                          type: str = 'index') -> str:
        """
        生成指数基本面数据的分析提示词
        
        参数:
            fundamental_frame: 包含基本面数据的DataFrame，字段包括：
                Stock, Date, PE_TTM, PB, PS_TTM, PCF_TTM, dividend_ratio
            type: 分析类型，可选值：
                - 'individual_stock': 个股分析
                - 'industry': 行业分析
                - 'index': 指数分析
        
        返回:
            str: 格式化的分析结果字符串
        """
        # 根据类型设置不同的描述文字
        type_descriptions = {
            'individual_stock': ('股票', '该股票'),
            'industry': ('行业', '该行业'),
            'index': ('指数', '该指数')
        }
        
        type_name, type_prefix = type_descriptions.get(type, ('股票', '该股票'))
        
        res_prompt = f"对于{type_name}{stock_name}（代码：{stock_code}），估值数据分析如下：\n\n"
        
        # 确保数据按日期排序
        fundamental_frame = fundamental_frame.sort_values('Date')
        
        # 定义指标及其中文名称
        indicators = {
            'PE_TTM': '市盈率',
            'PB': '市净率',
            'PS_TTM': '市销率',
            'PCF_TTM': '市现率',
            'dividend_ratio': '股息率%'
        }
        
        # 需要去除负值的指标
        negative_check_indicators = ['PE_TTM', 'PCF_TTM']
        
        for indicator_col, indicator_name in indicators.items():
            if indicator_col not in fundamental_frame.columns:
                res_prompt += f"【{indicator_name}】数据缺失\n\n"
                continue
            
            # 获取该指标的所有数据
            indicator_data = fundamental_frame[indicator_col].copy()
            
            # 获取最新值
            
            if indicator_col == 'dividend_ratio':
                indicator_data = indicator_data * 100
            latest_value = indicator_data.iloc[-1]
            # 对于市盈率和市现率，需要特殊处理负值
            if indicator_col in negative_check_indicators:
                # 检查最新值是否为负
                if pd.notna(latest_value) and latest_value < 0:
                    if indicator_col == 'PE_TTM':
                        if type == 'individual_stock':
                            res_prompt += f"【{indicator_name}】当前股票近一年处于亏损，此分析法失效\n\n"
                        elif type == 'industry':
                            res_prompt += f"【{indicator_name}】当前行业整体近一年处于亏损，此分析法失效\n\n"
                        elif type == 'index':
                            res_prompt += f"【{indicator_name}】当前指数成分股整体近一年处于亏损，此分析法失效\n\n"
                    elif indicator_col == 'PCF_TTM':
                        if type == 'individual_stock':
                            res_prompt += f"【{indicator_name}】当前股票近一年自由现金流为负，此分析法失效\n\n"
                        elif type == 'industry':
                            res_prompt += f"【{indicator_name}】当前行业整体近一年自由现金流为负，此分析法失效\n\n"
                        elif type == 'index':
                            res_prompt += f"【{indicator_name}】当前指数成分股整体近一年自由现金流为负，此分析法失效\n\n"
                    continue
                
                # 去除所有负值后进行分析
                indicator_data_cleaned = indicator_data[indicator_data > 0]
            else:
                # 其他指标不需要去除负值
                indicator_data_cleaned = indicator_data[pd.notna(indicator_data)]
            
            # 检查清洗后的数据是否足够
            if len(indicator_data_cleaned) == 0:
                res_prompt += f"【{indicator_name}】有效数据不足，无法进行分析\n\n"
                continue
            
            # 计算统计量
            median_value = indicator_data_cleaned.median()
            max_value = indicator_data_cleaned.max()
            min_value = indicator_data_cleaned.min()
            q25_value = indicator_data_cleaned.quantile(0.25)
            q75_value = indicator_data_cleaned.quantile(0.75)
            
            # 计算当前值在历史窗口中的位置（高于历史X%的时间）
            if pd.notna(latest_value):
                if indicator_col in negative_check_indicators:
                    # 对于去除负值的指标，只与正值比较
                    count_below = (indicator_data_cleaned < latest_value).sum()
                    percentile = (count_below / len(indicator_data_cleaned)) * 100
                else:
                    # 对于其他指标，与所有有效值比较
                    count_below = (indicator_data_cleaned < latest_value).sum()
                    percentile = (count_below / len(indicator_data_cleaned)) * 100
                
                # 生成描述（根据type调整描述文字）
                res_prompt += f"【{indicator_name}】\n"
                if indicator_col == 'dividend_ratio':
                    res_prompt += f"  当前值：{(latest_value):.2f}%\n"
                else:
                    res_prompt += f"  当前值：{latest_value:.2f}\n"
                
                # 根据类型生成不同的描述
                if type == 'individual_stock':
                    res_prompt += f"  {type_prefix}当前{indicator_name}为 {latest_value:.2f}，高于近{len(fundamental_frame)}个交易日中{percentile:.1f}%的时间\n"
                elif type == 'industry':
                    res_prompt += f"  {type_prefix}当前{indicator_name}为 {latest_value:.2f}，高于历史{len(fundamental_frame)}个交易日中{percentile:.1f}%的时间\n"
                elif type == 'index':
                    res_prompt += f"  {type_prefix}当前{indicator_name}为 {latest_value:.2f}，高于历史{len(fundamental_frame)}个交易日中{percentile:.1f}%的时间\n"
                
                res_prompt += f"  历史统计指标（基于{len(indicator_data_cleaned)}个有效数据点）：\n"
                res_prompt += f"    - 中位数：{median_value:.2f}\n"
                res_prompt += f"    - 最大值：{max_value:.2f}\n"
                res_prompt += f"    - 最小值：{min_value:.2f}\n"
                res_prompt += f"    - 25%分位值：{q25_value:.2f}\n"
                res_prompt += f"    - 75%分位值：{q75_value:.2f}\n"
            
            else:
                res_prompt += f"【{indicator_name}】当前值缺失\n"
            
            res_prompt += "\n"
        
        return res_prompt

    def get_analysis_prompt(self, stock_code: str, analysis_date: int) -> str:
        # Load data once
        daily_technical_data, weekly_technical_data, monthly_technical_data = self.stock_loader.load_benchmark_index_with_macd(analysis_date, look_back_period=5)
        index_fundamental = self.financial_loader.calculate_index_fundamentals(analysis_date)
        index_code = self.config.data.index_map[self.config.benchmark_index]
        recent_news = self.news_loader.load_news(index_code, analysis_date).sort_values(["Date"], ascending=False)
        
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
        res_day = self._generate_analysis_prompt_from_dataframe(daily_technical_data[daily_cols])
        res_week = self._generate_analysis_prompt_from_dataframe(weekly_technical_data[weekly_cols])
        res_month = self._generate_analysis_prompt_from_dataframe(monthly_technical_data[monthly_cols])
        fundamentals_str = self.generate_analysis_prompt_from_stock_fundamentals(index_code, self.config.benchmark_index, index_fundamental, 'index')
        news_str = self._generate_analysis_prompt_from_dataframe(recent_news)
        
        # Build prompt using list for efficient string concatenation
        prompt_parts = [
            """
        你是二级市场宏观和指数择时分析师.请分析基准指数{self.config.benchmark_index}在{analysis_date} 的技术面, 基本面 和消息面层面的信息.
        "你的回复结果应以json 格式呈现 包括如下内容 \n"
        "analysis_content: 分析内容 key_points: 分析要点 分析要点应该是一个字符串列表 从 消息面 技术面 基本面 和总结这四点来展开。在key_points 部分 你不要包含任何有关具体日期和具体指数点位的信息 
        risk_factors: 分析风险因素 investment_suggestion: 关于未来走势和择时仓位控制的建议. 
        你的仓位控制应基于指数当前估值位置 安全边际 与短期内对未来走势的判断决定. 如果你强烈认为指数低估 指标超卖存在反弹可能 你应输出满仓或者接近满仓的仓位控制建议. 
        如果你强烈认为指数高估 指标超买存在回调可能 你应输出空仓或者接近空仓的仓位控制建议. 
        "confidence_score: 分析置信度(0 - 100)\n"
        """,
            f"""
        {self.config.benchmark_index}在{analysis_date} 的日线级别基础行情信息和技术面指标信息如下 你应分析最近5个交易日的高开低收 成交量 与均线 布林线 RSI的关系 以及这些技术指标所揭示出来的信息：
        "字段含义为 Open: 开盘价 High: 最高价 Low: 最低价 Close: 收盘价 Value: 成交量
        MV_5_D: 5日均线 MV_10_D: 10日均线 MV_20_D: 20日均线 MV_60_D: 60日均线 MV_120_D: 120日均线 MV_250_D: 250日均线
        DIF_D: 差离值 DEA_D: 讯号线 MACD_D: 平滑移动平均线 BOLL_UPPER_D: 布林线上轨 BOLL_MID_D: 布林线中轨 BOLL_LOWER_D: 布林线下轨
        RSI_D: 相对强弱指数\n"
        "{res_day}"
        
        其周线级别基础行情信息和技术面指标信息如下：\n\n 你应分析最近5个交易周的高开低收 成交量 与均线 布林线 RSI的关系 以及这些技术指标所揭示出来的信息"
        "字段含义为 Open_W: 开盘价 High_W: 最高价 Low_W: 最低价 Close_W: 收盘价 Value_W: 成交量
        MV_5_W: 5周均线 MV_10_W: 10周均线 MV_20_W: 20周均线 MV_60_W: 60周均线 MV_120_W: 120周均线
        DIF_W: 差离值 DEA_W: 讯号线 MACD_W: 平滑移动平均线 BOLL_UPPER_W: 布林线上轨 BOLL_MID_W: 布林线中轨 BOLL_LOWER_W: 布林线下轨
        RSI_W: 相对强弱指数\n"
        "{res_week}"
        
        其月线级别基础行情信息和技术面指标信息如下：\n\n 你应分析最近5个交易月的高开低收 成交量 与均线 布林线 RSI的关系 以及这些技术指标所揭示出来的信息"
        "字段含义为 Open_M: 开盘价 High_M: 最高价 Low_M: 最低价 Close_M: 收盘价 Value_M: 成交量
        MV_5_M: 5月均线 MV_10_M: 10月均线 MV_20_M: 20月均线 MV_60_M: 60月均线
        DIF_M: 差离值 DEA_M: 讯号线 MACD_M: 平滑移动平均线 BOLL_UPPER_M: 布林线上轨 BOLL_MID_M: 布林线中轨 BOLL_LOWER_M: 布林线下轨
        RSI_M: 相对强弱指数\n"
        "{res_month}\n"
        """,
            f"""
        {self.config.benchmark_index}在{analysis_date} 的基本面数据如下:\n"
        """,
            fundamentals_str,
            f"""
        {self.config.benchmark_index}在{analysis_date} 的新闻数据如下:\n"
        "{news_str}"
        """
        ]
        
        return ''.join(prompt_parts)
