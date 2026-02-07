"""
分析师基类
Base analyst class for stock analysis
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from multi_agent_stock_selection.config.config import GlobalConfig
from multi_agent_stock_selection.utils.llm import OpenAIModel
from openai import OpenAI
import json
import re


logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class AnalystPattern:
    """
    分析师论点结果类
    """
    thesis_polar: bool
    thesis_content: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalystPattern':
        return cls(
            thesis_polar=data.get("thesis_polar"),
            thesis_content=data.get("thesis_content", "")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thesis_polar": self.thesis_polar,
            "thesis_content": self.thesis_content
        }


@dataclass
class AnalysisResult:
    """
    分析结果数据类
    """
    stock_code: str
    analysis_date: int
    analyst_type: str
    analysis_content: List[AnalystPattern]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stock_code": self.stock_code,
            "analysis_date": self.analysis_date,
            "analyst_type": self.analyst_type,
            "analysis_content": [p.to_dict() for p in self.analysis_content]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        raw_list = data.get("analysis_content", [])

        patterns = []
        for item in raw_list:
            if isinstance(item, AnalystPattern):
                patterns.append(item)
            elif isinstance(item, dict):
                patterns.append(AnalystPattern.from_dict(item))
            else:
                raise ValueError(f"Invalid analysis_content element: {item}")

        return cls(
            stock_code=data["stock_code"],
            analysis_date=data["analysis_date"],
            analyst_type=data["analyst_type"],
            analysis_content=patterns
        )


class BaseAnalyst(ABC):
    """
    分析师基类
    Base class for all analysts
    """
    
    def __init__(
        self, 
        llm_client: OpenAI,
        model: OpenAIModel,
        config: GlobalConfig,
        analyst_name: str,
        max_retry: int = 3,
    ):
        """
        初始化分析师
        
        Args:
            analyst_name: 分析师名称，如果未提供则使用类名
            llm_client: 大语言模型客户端
            data_processor: 数据处理器
            config: 配置参数
        """
        self.analyst_name = analyst_name 
        self.llm_client = llm_client
        self.model = model
        self.config = config or {}
        self.max_retry = max_retry
        
        # 分析结果缓存
        self._analysis_cache = {}

    def generate_system_prompt(self) -> str:
        """
        生成系统提示词
        
        Returns:
            系统提示词
        """
        default_system_prompt = """You are an expert on financial market analysis, portfolio management and quantitative trading. Strictly follow the user's requirements to generate sound, logical and detailed responses. Your ouput should only contain a JSON object.
                                Do not include any explanations, addditional text, notes, code block markers like ``` or ```json."""
        return default_system_prompt
        
    
    @abstractmethod
    def get_analysis_prompt(
        self, 
        stock_code: str, analysis_date: int, **kwargs
    ) -> str:
        """
        生成用户提示词
        
        Args:
            stock_code: 股票代码
            analysis_data: 分析数据
            **kwargs: 其他参数
            
        Returns:
            用户提示词
        """
        pass

    def analysis(self, stock_code: str, analysis_date: int, **kwargs) -> AnalysisResult:
        """
        分析股票，带重试机制
        """
        system_prompt = self.generate_system_prompt()
        user_prompt = self.get_analysis_prompt(stock_code, analysis_date, **kwargs)
        
        max_retries = self.max_retry
        json_res = None
        for attempt in range(max_retries):
            try:
                # 调用大模型
                res, _ = self.model.chat_generate(
                    client=self.llm_client, 
                    system_prompt=system_prompt, 
                    user_prompt=user_prompt
                )
                # print("----")
                # print(res)
                # print("----")
                
                # 尝试解析JSON
                json_res = json.loads(res)
                break
                
            except json.JSONDecodeError as e:
                logger.warning(f"股票 {stock_code} JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                pattern = re.compile(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```|\{([\s\S]*?)\}',re.MULTILINE)
                for match in pattern.finditer(res):
                    candidate = match.group(1) or f'{{{match.group(2)}}}'
                try:
                    json_res = json.loads(candidate)
                except (json.JSONDecodeError, AssertionError) as exc:
                    print(f"Failed to candidate {candidate}: {exc}")
                    if attempt == max_retries - 1:
                        logger.error(f"股票 {stock_code} JSON解析失败，已达到最大重试次数")
                        return None
                    continue
                    
            except Exception as e:
                logger.warning(f"股票 {stock_code} 大模型调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"股票 {stock_code} 分析失败，已达到最大重试次数: {e}")
                    return None
        
        analysis_type = self.analyst_name
        analysis_date = analysis_date
        # analysis_content = json_res["arguments"]

        result = AnalysisResult.from_dict({
            "stock_code": stock_code,
            "analysis_date": analysis_date,
            "analyst_type": analysis_type,
            "analysis_content": json_res["arguments"]
        })
        
        # Cache the result to JSON file
        cache_dir = os.path.join(self.config.data.analysis_res_path, analysis_type)
        cache_dir = os.path.join(cache_dir, self.config.benchmark_index)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{stock_code}_{analysis_date}.json")
        print(f"Cache file: {cache_file}")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        return result