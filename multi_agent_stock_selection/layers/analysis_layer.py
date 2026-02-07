"""
Analysis layer for multi-dimensional stock analysis using various analyst agents
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..agents.news_analyst import NewsAnalyst
from ..agents.technical_analyst import TechnicalAnalyst
from ..agents.fundamental_analyst import FundamentalAnalyst
from ..agents.macro_analyst import MacroAnalyst
from ..agents.base_analyst import AnalysisResult
from ..config.config import GlobalConfig
from openai import OpenAI
from ..utils.llm import OpenAIModel
from ..utils.non_blocking_thread_pool import NonBlockingThreadPool

logger = logging.getLogger(__name__)

class AnalysisLayer:
    def __init__(self, config: GlobalConfig, open_ai: OpenAI, open_ai_model: OpenAIModel):
        self.config = config
        self.stock_labels = pd.read_parquet(config.data.stock_labels_path)
        self.index_classification = pd.read_parquet(config.data.industry_path, columns=["Stock", "Date", "index_classification", "industry_name", "industry_code"], filters = [("index_classification", "==", "sw_l2")])
        self.news_analyst = NewsAnalyst(llm_client=open_ai, model=open_ai_model, config=config, stock_labels=self.stock_labels, index_classification=self.index_classification)
        self.technical_analyst = TechnicalAnalyst(llm_client=open_ai, model=open_ai_model, config=config, stock_labels=self.stock_labels, index_classification=self.index_classification)
        self.fundamental_analyst = FundamentalAnalyst(llm_client=open_ai, model=open_ai_model, config=config, stock_labels=self.stock_labels, index_classification=self.index_classification)
    

    def analyze(self, stock_codes: list[str], date: int, parrell: bool = True) -> tuple[Dict[str, AnalysisResult], Dict[str, AnalysisResult], Dict[str, AnalysisResult], Dict[str, AnalysisResult]]:
        news_results = {}
        technical_results = {}
        fundamental_results = {}
        macro_results = {}
        
        if parrell:
            return self._analyze_parallel(stock_codes, date, news_results, technical_results, fundamental_results, macro_results)
        else:
            return self._analyze_sequential(stock_codes, date, news_results, technical_results, fundamental_results, macro_results)

    def _analyze_parallel(self, stock_codes: List[str], date: int, news_results: Dict, technical_results: Dict, fundamental_results: Dict, macro_results: Dict) -> tuple:
        
        max_workers = min(len(stock_codes) * 3, 20) 
        thread_pool = NonBlockingThreadPool(max_workers=max_workers)
        
        all_tasks = []
        

        for stock_code in stock_codes:
            all_tasks.append({
                'type': 'news',
                'stock_code': stock_code,
                'analyst': self.news_analyst,
                'date': date
            })
            all_tasks.append({
                'type': 'technical',
                'stock_code': stock_code,
                'analyst': self.technical_analyst,
                'date': date
            })
            
            # 基本面分析任务
            all_tasks.append({
                'type': 'fundamental',
                'stock_code': stock_code,
                'analyst': self.fundamental_analyst,
                'date': date
            })
        
        def analyze_task(task_info):
            try:
                result = task_info['analyst'].analysis(task_info['stock_code'], task_info['date'])
                return {
                    'type': task_info['type'],
                    'stock_code': task_info['stock_code'],
                    'result': result,
                    'success': True
                }
            except Exception as e:
                logger.error(f"分析任务失败: {task_info['type']} - {task_info['stock_code']}: {e}")
                return {
                    'type': task_info['type'],
                    'stock_code': task_info['stock_code'],
                    'result': None,
                    'success': False,
                    'error': str(e)
                }
        
        def result_callback(result, original_task):
            """处理分析结果的回调函数"""
        thread_pool.map_and_collect_results(
            iterable=all_tasks,
            task_func=analyze_task,
            result_callback=result_callback,
            progress_desc="股票分析进度",
            monitor_interval=150
        )

    def _analyze_sequential(self, stock_codes: List[str], date: int, news_results: Dict, technical_results: Dict, fundamental_results: Dict, macro_results: Dict) -> tuple:
        logger.info(f"开始串行分析 {len(stock_codes)} 只股票，分析日期: {date}")
        
        for i, stock_code in tqdm(enumerate(stock_codes), desc = f"analysis stocks sequentially on {date}"):
            logger.info(f"分析股票 {stock_code} ({i+1}/{len(stock_codes)})")
            
            try:
                # 消息面分析
                news_result = self.news_analyst.analysis(stock_code, date)
                if news_result:
                    news_results[stock_code] = news_result
                    logger.info(f"股票 {stock_code} 消息面分析完成")
                else:
                    logger.warning(f"股票 {stock_code} 消息面分析失败")
                    
            except Exception as e:
                logger.error(f"股票 {stock_code} 消息面分析异常: {e}")
                
            try:
                # 技术面分析
                technical_result = self.technical_analyst.analysis(stock_code, date)
                if technical_result:
                    technical_results[stock_code] = technical_result
                    logger.info(f"股票 {stock_code} 技术面分析完成")
                else:
                    logger.warning(f"股票 {stock_code} 技术面分析失败")
                    
            except Exception as e:
                logger.error(f"股票 {stock_code} 技术面分析异常: {e}")
                
            try:
                # 基本面分析
                fundamental_result = self.fundamental_analyst.analysis(stock_code, date)
                if fundamental_result:
                    fundamental_results[stock_code] = fundamental_result
                    logger.info(f"股票 {stock_code} 基本面分析完成")
                else:
                    logger.warning(f"股票 {stock_code} 基本面分析失败")
                    
            except Exception as e:
                logger.error(f"股票 {stock_code} 基本面分析异常: {e}")
        
        # 宏观分析（对整个市场/指数进行分析）
        try:
            logger.info("开始宏观分析")
            # 使用config中的index_map获取基准指数代码
            benchmark_code = self.config.data.index_map.get(self.config.benchmark_index)
            if benchmark_code:
                macro_result = self.macro_analyst.analysis(benchmark_code, date)
                if macro_result:
                    macro_results[benchmark_code] = macro_result
                    logger.info("宏观分析完成")
                else:
                    logger.warning("宏观分析失败")
            else:
                logger.error(f"未找到基准指数 {self.config.benchmark_index} 对应的代码")
                
        except Exception as e:
            logger.error(f"宏观分析异常: {e}")
        
        logger.info(f"串行分析完成。消息面: {len(news_results)}, 技术面: {len(technical_results)}, 基本面: {len(fundamental_results)}, 宏观: {len(macro_results)}")
        
        return news_results, technical_results, fundamental_results, macro_results
