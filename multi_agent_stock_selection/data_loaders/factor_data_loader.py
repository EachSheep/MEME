"""
Factor data loader for screening factors.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from ..utils.date_utils import get_begin_date

logger = logging.getLogger(__name__)

class FactorDataLoader:
    def __init__(self, config, index: str):
        self.config = config
        self._cache = {}
        self.index = index
        self.factor_data = pd.read_parquet(self.config.all_features_path)
        self.index_pool = pd.read_parquet(os.path.join(self.config.data_root, f"{self.index}.parq"))
        self.factor_data = self.factor_data.merge(self.index_pool[["Stock","Date"]], on=["Stock", "Date"], how="inner").sort_values(["Stock", "Date"])
        features = np.setdiff1d(self.factor_data.columns, ["Stock", "Date"])
        for fea in features:
            if fea not in ["QTLD60", "VSUMN_R60", "dividend_ratio_refined","ROETTM_stability",
                'EP_refined','SP_refined','BP_refined']:
                self.factor_data[fea] = - self.factor_data[fea]
            self.factor_data[fea] = self.factor_data.groupby("Date")[fea].transform(lambda x : (x - x.mean()) / x.std())
        self.factor_data["sum"] = self.factor_data[features].sum(axis=1)

    
    def load_factor_data(self, date: int) -> pd.DataFrame:
        return self.factor_data[self.factor_data['Date'] == date][["Stock", "Date", "sum"]]
    
