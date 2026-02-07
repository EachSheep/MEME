"""
基本面和行业综合分析师
Fundamental and industry analyst for comprehensive fundamental analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from openai import OpenAI
import logging

from .base_analyst import BaseAnalyst
from ..data_loaders.stock_data_loader import StockDataLoader
from ..data_loaders.financial_data_loader import FinancialDataLoader
from ..utils.llm import OpenAIModel
from ..config.config import GlobalConfig
from ..utils.prompt import PROMPT_FUNDAMENTALS, PROMPT_INSTRUCTIONS
logger = logging.getLogger(__name__)

class FundamentalAnalyst(BaseAnalyst):
    """
    基本面分析师
    """
    def __init__(self, llm_client: OpenAI, model: OpenAIModel, config: GlobalConfig=None, stock_labels: pd.DataFrame = None, index_classification: pd.DataFrame = None):
        
        super().__init__(llm_client=llm_client, model=model, config=config, analyst_name= self.__class__.__name__)
        self.stock_loader = StockDataLoader(config, stock_labels=stock_labels, index_classification=index_classification)
        self.financial_loader = FinancialDataLoader(config, index_classification=index_classification)
    
    def _generate_analysis_prompt_from_dataframe(self, data: pd.DataFrame) -> str:
        return data.to_string(index=False)
    
    def generate_analysis_prompt_from_stock_fundamentals(self, stock_code: str, stock_name: str, 
                                                          fundamental_frame: pd.DataFrame,
                                                          type: str = 'individual_stock') -> str:
        """
        生成基于基本面数据的分析提示词
        
        参数:
            stock_code: 股票/行业/指数代码
            stock_name: 股票/行业/指数名称
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
        
        # 确保数据按日期排序
        fundamental_frame = fundamental_frame.sort_values('Date')
        
        # 预计算重复使用的值
        frame_length = len(fundamental_frame)
        
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
        
        # 预定义不同类型的错误消息模板
        error_messages = {
            'PE_TTM': {
                'individual_stock': "当前股票近一年处于亏损，此分析法失效",
                'industry': "当前行业整体近一年处于亏损，此分析法失效",
                'index': "当前指数成分股整体近一年处于亏损，此分析法失效"
            },
            'PCF_TTM': {
                'individual_stock': "当前股票近一年自由现金流为负，此分析法失效",
                'industry': "当前行业整体近一年自由现金流为负，此分析法失效",
                'index': "当前指数成分股整体近一年自由现金流为负，此分析法失效"
            }
        }
        
        # 预定义类型描述模板
        type_description_templates = {
            'individual_stock': f"  {type_prefix}当前{{indicator_name}}为 {{latest_value:.2f}}，高于近{frame_length}个交易日中{{percentile:.1f}}%的时间\n",
            'industry': f"  {type_prefix}当前{{indicator_name}}为 {{latest_value:.2f}}，高于历史{frame_length}个交易日中{{percentile:.1f}}%的时间\n",
            'index': f"  {type_prefix}当前{{indicator_name}}为 {{latest_value:.2f}}，高于历史{frame_length}个交易日中{{percentile:.1f}}%的时间\n"
        }
        
        # 使用列表收集字符串片段
        prompt_parts = [f"对于{type_name}{stock_name}（代码：{stock_code}），估值数据分析如下：\n\n"]
        
        for indicator_col, indicator_name in indicators.items():
            if indicator_col not in fundamental_frame.columns:
                prompt_parts.append(f"【{indicator_name}】数据缺失\n\n")
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
                    # 只有当type在错误消息字典中存在时才输出错误消息
                    if type in error_messages[indicator_col]:
                        error_msg = error_messages[indicator_col][type]
                        prompt_parts.append(f"【{indicator_name}】{error_msg}\n\n")
                    continue
                
                # 去除所有负值后进行分析
                indicator_data_cleaned = indicator_data[indicator_data > 0]
            else:
                # 其他指标不需要去除负值
                indicator_data_cleaned = indicator_data[pd.notna(indicator_data)]
            
            # 检查清洗后的数据是否足够
            if len(indicator_data_cleaned) == 0:
                prompt_parts.append(f"【{indicator_name}】有效数据不足，无法进行分析\n\n")
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
                indicator_parts = [f"【{indicator_name}】\n"]
                
                if indicator_col == 'dividend_ratio':
                    indicator_parts.append(f"  当前值：{latest_value:.2f}%\n")
                else:
                    indicator_parts.append(f"  当前值：{latest_value:.2f}\n")
                
                # 根据类型生成不同的描述
                template = type_description_templates.get(type, type_description_templates['individual_stock'])
                indicator_parts.append(template.format(
                    indicator_name=indicator_name, 
                    latest_value=latest_value, 
                    percentile=percentile
                ))
                
                indicator_parts.extend([
                    f"  历史统计指标（基于{len(indicator_data_cleaned)}个有效数据点）：\n",
                    f"    - 中位数：{median_value:.2f}\n",
                    f"    - 最大值：{max_value:.2f}\n",
                    f"    - 最小值：{min_value:.2f}\n",
                    f"    - 25%分位值：{q25_value:.2f}\n",
                    f"    - 75%分位值：{q75_value:.2f}\n"
                ])
                
                prompt_parts.extend(indicator_parts)
            else:
                prompt_parts.append(f"【{indicator_name}】当前值缺失\n")
            
            prompt_parts.append("\n")
        
        return ''.join(prompt_parts)

    
    def generate_analysis_prompt(self, code: str, name: str, industry_name: str,
                                code_fundamental_analysis: str,industry_fundamental_analysis: str,
                                code_income_analysis: str, code_balance_analysis: str,
                                industry_income_analysis: str, industry_balance_analysis: str,
                                basic_info: dict[str, Any]) -> str:
        # Build prompt using list for efficient string concatenation
        prompt_parts = [
            f"{PROMPT_FUNDAMENTALS}\n",
            f'对于股票{name}（代码：{code}），其个股与所在{industry_name}行业基本面数据分析如下：\n\n',
            f'个股：\n\n',
            code_fundamental_analysis,
            f'\n {name} 最近五个财报季 利润表指标如下: 你可以根据公司最近的财报表现 分析公司的盈利能力以及成长能力 以及目前公司的估值水平是否与财务状况匹配\n\n',
            code_income_analysis,
            f'\n {name} 最近五个财报季 资产负债表指标如下: 你可以根据公司最近的财报表现 分析公司的家底状况 以及运营状况. 为防止绝对数据干扰(举例: 由于公司现金储备绝对含量高而偏向该公司) 公司最新一个交易日的市值为 {basic_info["mv"]} 亿元\n\n',
            code_balance_analysis,
            f'\n 行业：\n\n',
            industry_fundamental_analysis,
            f'\n {industry_name} 行业最近五个财报季 利润表指标如下\n\n',
            industry_income_analysis,
            f'\n {industry_name} 行业最近五个财报季 资产负债表指标如下\n\n',
            industry_balance_analysis,
            f"{PROMPT_INSTRUCTIONS}\n"
        ]
        
        return ''.join(prompt_parts)
    

    def get_analysis_prompt(self, stock_code: str, analysis_date: int, **kwargs) -> str:
        """
        获取基本面分析数据
        """
        try:
            # Load all data first
            basic_info = self.stock_loader.load_stock_basic_info(stock_code, analysis_date)
            if len(basic_info.keys()) == 0:
                return {"error": "无法获取股票基础信息"}
            
            code_income, industry_income = self.financial_loader.load_latest_income_statement(stock_code, analysis_date)
            if code_income.empty or industry_income.empty:
                return {"error": "无法获取利润表数据"}
            
            code_balance, industry_balance = self.financial_loader.load_latest_balance_sheet(stock_code, analysis_date)
            if code_balance.empty or industry_balance.empty:
                return {"error": "无法获取资产负债表数据"}
            
            code_fundamental, industry_fundamental, industry_code, industry_name = self.financial_loader.calculate_stock_fundamentals(stock_code, analysis_date)
            
            # Pre-compute all DataFrame conversions to strings
            code_fundamental_analysis = self.generate_analysis_prompt_from_stock_fundamentals(stock_code,
                                                                                            basic_info['stock_name'],
                                                                                            code_fundamental, 'individual_stock')
            industry_fundamental_analysis = self.generate_analysis_prompt_from_stock_fundamentals(industry_code, industry_name, industry_fundamental, 'industry')
            code_income_analysis = self._generate_analysis_prompt_from_dataframe(code_income)
            industry_income_analysis = self._generate_analysis_prompt_from_dataframe(industry_income)
            code_balance_analysis = self._generate_analysis_prompt_from_dataframe(code_balance)
            industry_balance_analysis = self._generate_analysis_prompt_from_dataframe(industry_balance)
            
            # Generate final prompt
            res_prompt = self.generate_analysis_prompt(stock_code, basic_info['stock_name'], industry_name,
                                                       code_fundamental_analysis, industry_fundamental_analysis,
                                                       code_income_analysis, code_balance_analysis,
                                                       industry_income_analysis, industry_balance_analysis,
                                                       basic_info)
            return res_prompt
        except Exception as e:
            return {"error": str(e)}
