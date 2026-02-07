"""
Financial data loader for income statement and balance sheet data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import os

from multi_agent_stock_selection.config.config import GlobalConfig
from multi_agent_stock_selection.utils.date_utils import get_begin_date

logger = logging.getLogger(__name__)

class FinancialDataLoader:
    
    # 提取重复的rename字典为类常量
    INCOME_RENAME_DICT = {
        "Stock": "股票代码", "ReportPeriod": "报告期",
        "ROETTM": "净资产收益率", "gross_margin": "单季度毛利率", 
        "gross_margin_TTM": "毛利率TTM", "net_margin": "单季度净利率", 
        "net_margin_TTM": "净利率TTM", 
        "net_profit_parent_growth_rate": "单季度归母净利润增长率", 
        "net_profit_parent_TTM_growth_rate": "归母净利润增长率TTM",
        "operating_revenue_growth_rate": "单季度营业收入增长率", 
        "operating_revenue_TTM_growth_rate": "营业收入增长率TTM"
    }
    
    def __init__(self, data_config: GlobalConfig, used_index: str = "sw_l2", index_classification: pd.DataFrame = None):
        """
        初始化财务数据加载器
        
        Args:
            data_config: 数据配置对象
        """
        self.config = data_config
        self.income = pd.read_parquet(self.config.data.income_path)
        self.balance = pd.read_parquet(self.config.data.balance_path)
        # self.pre_process_financial_data()
        self.fundamental_data = pd.read_parquet(self.config.data.fundamental_data_path)
        self.industry = index_classification
        self.base_data = pd.read_parquet(self.config.data.base_data_path, columns=["Stock", "Date", "FREE_MV", "MV"])
        self.used_index = used_index
        self.index_pool = pd.read_parquet(self.config.data.index_pool_path)
        self._cache = {}
    
    def pre_process_financial_data(self) -> pd.DataFrame:
        """
        预处理财务数据, 对于所有财务数据, 涉及大小在亿元及以上的， / 1e8, 并保留三位小数，提升可读性
        """
        modified_income_features = ['operating_revenue',
       'operating_cost', 'total_operating_expenses', 'operating_profit',
       'selling_expenses', 'admin_expenses', 'financial_expenses',
       'rd_expenses', 'net_profit_parent', 'net_profit_excl_non_recurring',
       'operating_revenue_TTM_growth_rate', 'operating_cost_TTM', 'total_operating_expenses_TTM',
       'operating_profit_growth_rate',
       'operating_profit_TTM_growth_rate', 
       'selling_expenses_TTM','admin_expenses_TTM',
       'financial_expenses_TTM', 'rd_expenses_TTM','net_profit_parent_TTM',
       'net_profit_excl_non_recurring_TTM']

        modified_balance_features = ['cash_and_equivalents',
       'total_liab', 'total_assets', 'net_assets', 'short_term_borrowings',
       'long_term_borrowings', 'contract_liabilities', 'accounts_receivable',
       'inventories', 'construction_in_progess']
        for key in modified_income_features:
            self.income[key] = (self.income[key] / 1e8).round(3)
        for key in modified_balance_features:
            self.balance[key] = (self.balance[key] / 1e8).round(3) 

    
    def load_latest_income_statement(
        self, 
        stock_code: str, 
        as_of_date: int,
        look_back_period: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载最新利润表数据
        
        Args:
            stock_code: 股票代码
            as_of_date: 截止日期
            
        Returns:
            最新利润表数据
        """
        try:
            used_compare_features = ["ROETTM", "gross_margin", "gross_margin_TTM", "net_margin", "net_margin_TTM", "net_profit_parent_growth_rate", "net_profit_parent_TTM_growth_rate",
            "operating_revenue_growth_rate", "operating_revenue_TTM_growth_rate"]
            
            # 查询行业代码
            industry_filtered = self.industry[
                (self.industry['Stock'] == stock_code) & 
                (self.industry['Date'] <= as_of_date) & 
                (self.industry['index_classification'] == self.used_index)
            ]
            industry_code = industry_filtered['industry_code'].iloc[-1]
            
            # 获取同行业股票列表（避免tolist，直接使用numpy array）
            same_industry_mask = (self.industry['industry_code'] == industry_code) & (self.industry['Date'] <= as_of_date)
            same_industry_stocks = self.industry.loc[same_industry_mask, 'Stock'].unique()
            
            # 筛选股票和发布日期
            stock_income = self.income[
                (self.income['Stock'] == stock_code) & 
                (self.income['ReleaseDate'] <= as_of_date)
            ].tail(look_back_period)[used_compare_features].copy()
            
            stock_income = stock_income.rename(columns=self.INCOME_RENAME_DICT)
            
            # 优化：预先计算日期边界，减少函数调用
            report_period_cutoff = get_begin_date(as_of_date, 0, 0, 15)
            
            # 优化：分离掩码计算后合并
            stock_mask = self.income['Stock'].isin(same_industry_stocks)
            period_mask = self.income['ReportPeriod'] <= report_period_cutoff
            same_industry_stocks_income = self.income.loc[stock_mask & period_mask, ["Stock", "ReportPeriod"] + used_compare_features]
            
            # 使用更高效的聚合方式（sort=False避免不必要排序）
            industry_income = (same_industry_stocks_income
                             .groupby("ReportPeriod", as_index=False, sort=False)
                             .mean(numeric_only=True)
                             .sort_values("ReportPeriod")
                             .tail(look_back_period))
            
            industry_income = industry_income.rename(columns=self.INCOME_RENAME_DICT)
            return stock_income, industry_income
        except Exception as e:
            logger.error(f"加载利润表数据失败: {stock_code}, {e}")
            return {}
    
    def load_latest_balance_sheet(
        self, 
        stock_code: str, 
        as_of_date: int,
        look_back_period: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载最新资产负债表数据
        
        Args:
            stock_code: 股票代码
            as_of_date: 截止日期
            
        Returns:
            最新资产负债表数据
        """
        try:
            used_compared_features = ["debt_to_asset_ratio"]
            stock_balance_sheet = self.balance[(self.balance['Stock'] == stock_code) & 
            (self.balance['ReleaseDate'] <= as_of_date)].sort_values('ReleaseDate').tail(look_back_period).copy()
            industry_code = self.industry[(self.industry['Stock'] == stock_code) & (self.industry['Date'] <= as_of_date) & (self.industry['index_classification'] == self.used_index)]['industry_code'].iloc[-1]
            same_industry_stocks = self.industry[(self.industry['industry_code'] == industry_code) & (self.industry['Date'] <= as_of_date)]['Stock'].unique().tolist()
            same_industry_stocks_balance_sheet = self.balance[self.balance['Stock'].isin(same_industry_stocks) & (self.balance['ReleaseDate'] <= get_begin_date(as_of_date, 0, 0, 15))][["Stock", "ReleaseDate"] + used_compared_features]
            industry_balance_sheet = same_industry_stocks_balance_sheet.groupby("ReleaseDate", as_index=False).mean(numeric_only=True).sort_values("ReleaseDate").tail(look_back_period)
            stock_balance_sheet = stock_balance_sheet.rename(columns={"Stock": "股票代码", "ReleaseDate": "报告期", "debt_to_asset_ratio": "资产负债率",
            'cash_and_equivalents': "现金及等价物",'total_liab': "总负债",'total_assets': "总资产(亿元)",
            'net_assets': "净资产",'short_term_borrowings': "短期借款(亿元)",'long_term_borrowings': "长期借款(亿元)",
            'contract_liabilities': "合同负债(亿元)",'accounts_receivable': "应收账款(亿元)",'inventories': "存货(亿元)",
            'debt_to_asset_ratio': "资产负债率", 'construction_in_progess': "在建工程(亿元)"})
            industry_balance_sheet = industry_balance_sheet.rename(columns={"Stock": "股票代码", "ReleaseDate": "报告期", "debt_to_asset_ratio": "资产负债率"})
            return stock_balance_sheet, industry_balance_sheet
        except Exception as e:
            logger.error(f"加载资产负债表数据失败: {stock_code}, {e}")
            return {}

    @staticmethod
    def calc_industry_metrics(
        df: pd.DataFrame,
        negative_placeholder: float = -1.0,
    ):
        import numpy as np

        out = {}

        def aggregate_check_and_compute(df_all, col, use_mv="MV"):
            # 列存在性快速检查
            if col not in df_all.columns or use_mv not in df_all.columns:
                return np.nan

            # 取底层 ndarray，避免 copy
            s = df_all[col].to_numpy(dtype="float64", copy=False)
            mv = df_all[use_mv].to_numpy(dtype="float64", copy=False)

            # 权重中的 NaN 视为 0（更稳健），避免 sum 变 NaN
            if mv is None:
                return np.nan
            mv = np.nan_to_num(mv, nan=0.0)

            # 非 NaN 掩码（指标为 NaN 的样本不参与）
            mask_non_nan = ~np.isnan(s)
            if not mask_non_nan.any():
                return np.nan

            # ---- Step 1: 行业合并"有无意义"检查（不剔除负值）----
            # 分母 = sum( MV / 指标 )
            denom_all = np.divide(mv[mask_non_nan], s[mask_non_nan]).sum()
            if denom_all <= 0:
                return negative_placeholder

            # ---- Step 2: 剔除负值，计算最终行业口径 ----
            mask_pos = mask_non_nan & (s > 0)
            if not mask_pos.any():
                return negative_placeholder

            numerator = mv[mask_pos].sum()
            denominator = np.divide(mv[mask_pos], s[mask_pos]).sum()
            return numerator / denominator if denominator > 0 else negative_placeholder

        def weighted_avg(df_all, col, use_mv="FREE_MV"):
            if col not in df_all.columns or use_mv not in df_all.columns:
                return np.nan

            s = df_all[col].to_numpy(dtype="float64", copy=False)
            w = df_all[use_mv].to_numpy(dtype="float64", copy=False)
            if s is None or w is None:
                return np.nan

            # 权重中的 NaN 视为 0；指标 NaN 直接剔除
            w = np.nan_to_num(w, nan=0.0)
            mask = (~np.isnan(s)) & (w > 0)

            if not mask.any():
                return np.nan

            wsum = w[mask].sum()
            if wsum <= 0:
                return np.nan

            return (s[mask] * w[mask]).sum() / wsum

        # 估值指标（总市值加权，整体法）
        if "PE_TTM" in df.columns:
            out["PE_TTM"] = aggregate_check_and_compute(df, "PE_TTM", use_mv="MV")
        if "PB" in df.columns:
            out["PB"] = aggregate_check_and_compute(df, "PB", use_mv="MV")
        # 修复：检查与使用同为 PS_TTM
        if "PS_TTM" in df.columns:
            out["PS_TTM"] = aggregate_check_and_compute(df, "PS_TTM", use_mv="MV")
        if "PCF_TTM" in df.columns:
            out["PCF_TTM"] = aggregate_check_and_compute(df, "PCF_TTM", use_mv="MV")

        # 比例类（流通市值加权平均）
        if "dividend_ratio" in df.columns:
            out["dividend_ratio"] = weighted_avg(df, "dividend_ratio", use_mv="FREE_MV")

        return pd.Series(out)
    
    def calculate_index_fundamentals(self, as_of_date: int, look_back_period: int = 2500) -> pd.DataFrame:
        """
        计算基准指数基本面数据.
        """
        if not os.path.exists(os.path.join(self.config.data.index_fundamental_path, f"{self.config.benchmark_index}.parq")):
            index_fundamental = (self.fundamental_data.merge(self.index_pool[["Stock", "Date"]], on=["Stock", "Date"], how="inner").
                                 merge(self.base_data[["Stock", "Date", "FREE_MV", "MV"]], on=["Stock", "Date"], how="inner").
                                 groupby("Date").
                                 apply(self.calc_industry_metrics).
                                 reset_index().sort_values("Date"))
            index_fundamental.to_parquet(os.path.join(self.config.data.index_fundamental_path, f"{self.config.benchmark_index}.parq"))
        else:
            index_fundamental = pd.read_parquet(os.path.join(self.config.data.index_fundamental_path, f"{self.config.benchmark_index}.parq"))
        sub_fundamental = index_fundamental[index_fundamental['Date'] <= as_of_date].sort_values('Date').tail(look_back_period)
        return sub_fundamental


    def calculate_stock_fundamentals(self, stock_code: str, as_of_date: int, look_back_period: int = 1250) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
        """
        计算股票和行业基本面数据. 返回给定窗口内 股票和行业的市盈率 市净率 市销率 市现率 股息率.
        优化版本：减少重复查询、降低内存使用、提升排序和筛选效率
        """
        # 优化1: 一次性获取股票行业信息，避免重复查询
        stock_industry_mask = (
            (self.industry['Stock'] == stock_code) &
            (self.industry['Date'] <= as_of_date) &
            (self.industry['index_classification'] == self.used_index)
        )
        stock_industry_info = self.industry.loc[stock_industry_mask].iloc[-1]
        industry_code = stock_industry_info['industry_code']
        industry_name = stock_industry_info['industry_name']
        
        # 优化2: 使用更高效的查询和切片操作
        # 避免对整个数据集排序，直接使用索引切片
        stock_mask = (self.fundamental_data['Stock'] == stock_code) & (self.fundamental_data['Date'] <= as_of_date)
        stock_fundamental_indices = self.fundamental_data.index[stock_mask]
        
        if len(stock_fundamental_indices) > look_back_period:
            # 只取最后N条记录的索引，避免排序整个数据集
            selected_indices = stock_fundamental_indices[-look_back_period:]
        else:
            selected_indices = stock_fundamental_indices
            
        current_stock_fundamental = self.fundamental_data.loc[selected_indices]
        
        # 优化3: 高效的merge操作 - 预筛选base_data
        base_data_mask = (
            (self.base_data['Stock'] == stock_code) &
            (self.base_data['Date'] <= as_of_date)
        )
        relevant_base_data = self.base_data.loc[base_data_mask, ["Stock", "Date", "FREE_MV"]]
        
        # 使用内存效率更高的merge操作
        current_stock_fundamental = current_stock_fundamental.merge(
            relevant_base_data, on=["Stock", "Date"], how="inner", copy=False
        )

        # 优化4: 高效的行业基本面数据处理
        industry_file_path = os.path.join(
            self.config.data.industry_fundamental_path, 
            f"{industry_code}_fundamental.parq"
        )
        
        if not os.path.exists(industry_file_path):
            # 优化5: 高效计算行业基本面数据
            # 预筛选同行业股票，避免不必要的copy操作
            same_industry_mask = (
                (self.industry['industry_code'] == industry_code) &
                (self.industry['index_classification'] == self.used_index)
            )
            same_industry_stocks = self.industry.loc[same_industry_mask, ["Stock", "Date"]]
            
            # 使用更高效的链式merge操作，避免多次数据复制
            same_industry_stocks_fundamental = (
                self.fundamental_data
                .merge(same_industry_stocks, on=["Stock", "Date"], how="inner", copy=False)
                .merge(
                    self.base_data[["Stock", "Date", "MV", "FREE_MV"]], 
                    on=["Stock", "Date"], 
                    how="inner", 
                    copy=False
                )
            )
            
            # 使用更高效的groupby操作
            industry_fundamental = (
                same_industry_stocks_fundamental
                .groupby("Date", sort=False)  # sort=False 避免不必要的排序
                .apply(self.calc_industry_metrics)
                .reset_index()
                .sort_values("Date")
            )
            
            # 确保目录存在并保存文件
            os.makedirs(os.path.dirname(industry_file_path), exist_ok=True)
            industry_fundamental.to_parquet(industry_file_path)
        else:
            industry_fundamental = pd.read_parquet(industry_file_path)
        
        # 优化6: 使用向量化操作和高效切片
        # 避免创建临时布尔mask，直接使用查询
        date_mask = industry_fundamental['Date'] <= as_of_date
        if date_mask.any():
            industry_fundamental_filtered = industry_fundamental.loc[date_mask]
            
            # 使用iloc进行高效的尾部切片
            if len(industry_fundamental_filtered) > look_back_period:
                industry_fundamental = industry_fundamental_filtered.iloc[-look_back_period:]
            else:
                industry_fundamental = industry_fundamental_filtered
        else:
            # 如果没有符合条件的数据，返回空DataFrame但保持结构
            industry_fundamental = industry_fundamental.iloc[0:0]

        return current_stock_fundamental, industry_fundamental, industry_code, industry_name


