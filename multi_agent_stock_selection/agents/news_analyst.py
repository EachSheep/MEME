"""
新闻分析师
News analyst for analyzing news sentiment and market impact
"""

import re
from akshare import stock_zh_ah_name
import pandas as pd
from typing import Dict, List, Any
from bs4 import BeautifulSoup
import logging
from openai import OpenAI
from ..utils.llm import OpenAIModel
from ..config.config import GlobalConfig
from ..data_loaders.news_data_loader import NewsDataLoader
from ..data_loaders.stock_data_loader import StockDataLoader

from .base_analyst import BaseAnalyst
from ..utils.prompt import PROMPT_NEWS, PROMPT_INSTRUCTIONS

logger = logging.getLogger(__name__)

class NewsAnalyst(BaseAnalyst):
    """
    新闻分析师
    """
    
    def __init__(self, llm_client: OpenAI, model: OpenAIModel, config: GlobalConfig=None, use_content: bool = False, stock_labels: pd.DataFrame = None,
    index_classification: pd.DataFrame = None):
        super().__init__(llm_client = llm_client, model = model, config = config, analyst_name= self.__class__.__name__)
        self.stock_loader = StockDataLoader(config, index=config.benchmark_index, stock_labels=None, index_classification=None)
        self.news_loader = NewsDataLoader(config = config, use_content = use_content)
    

    def _generate_analysis_prompt_from_dataframe(self, data: pd.DataFrame) -> str:
        return data.to_string(index=False)

    def get_analysis_prompt(self, stock_code: str, analysis_date: int, **kwargs) -> str:
        stock_basic_info = self.stock_loader.load_stock_basic_info(stock_code, analysis_date)  
        stock_name = stock_basic_info["stock_name"]                                                                                    
        recent_news = self.news_loader.load_news(stock_code, analysis_date).sort_values(["Date"], ascending=False)
        analysis_prompt = f"""
        {PROMPT_NEWS.format(stock_name=stock_name, stock_code=stock_code, look_back_period=self.news_loader.look_back_period)}\n
        新闻内容如下:
        {self._generate_analysis_prompt_from_dataframe(recent_news)} \n
        {PROMPT_INSTRUCTIONS}
        """
        # analysis_prompt = f"""你是二级市场新闻文本分析师.请分析股票{stock_name}（代码：{stock_code}）在{analysis_date} 近{self.news_loader.look_back_period} 的新闻内容信息 
        # 判断这些新闻是否可能影响到, 以及如何影响到 {stock_name}未来 在二级市场的未来走势. 
        # ,你的回复应以json 格式呈现 包括如下内容 \n"
        # "analysis_content: 分析内容 key_points: 字符串列表 一一总结可能对未来股价造成影响的新闻 以及可能对股价造成的影响 risk_factors: 字符串列表 从消息面角度，分析该股票的风险因素 
        # "\n
        #  新闻内容如下：
        #  {self._generate_analysis_prompt_from_dataframe(recent_news)}
        # \n"""
        return analysis_prompt
    