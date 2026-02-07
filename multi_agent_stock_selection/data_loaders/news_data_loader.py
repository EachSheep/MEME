"""
News data loader for news data
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from ..utils.date_utils import get_begin_date
from ..config.config import GlobalConfig

logger = logging.getLogger(__name__)

class NewsDataLoader:
    def __init__(self, config: GlobalConfig, use_content: bool = False, look_back_period: int = 3,
                use_index: bool = False):
        self.config = config
        if not use_index:
            self.news_path = self.config.data.news_path
        else:
            self.news_path = self.config.data.index_news_path
        self.use_content = use_content
        self.look_back_period = look_back_period
        self._cache = {}
        
    def load_news(self, stock_code: str, as_of_date: int) -> pd.DataFrame:
        """
        加载新闻数据
        """
        news_path = os.path.join(self.news_path, f"{stock_code}.parq")
        news = pd.read_parquet(news_path, columns=["Date", "NewsTitle"])
        if self.use_content:
            recent_news = news[(news['Date'] <= as_of_date) & (news['Date'] >= get_begin_date(as_of_date, 0, 0, self.look_back_period))][["Date","NewsTitle", "NewsContent"]].copy()
        else:
            recent_news = news[(news['Date'] <= as_of_date) & (news['Date'] >= get_begin_date(as_of_date, 0, 0, self.look_back_period))][["Date","NewsTitle"]].copy()
        return recent_news