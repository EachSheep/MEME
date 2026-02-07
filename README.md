<div align=center>

<h1>MEME: Modeling the evolutionary modes of financial markets</h1>

<!-- <img src="https://img.shields.io/badge/License-MIT-blue" alt="license"> -->

<div>
      <a href="https://https://scholar.google.com/citations?user=jk4YCOgAAAAJ&hl=zh-CN//" target="_blank">Taian Guo</a><sup>1,2</sup>,
      <a href="https://eachsheep.space/" target="_blank">Haiyang Shen</a><sup>*1</sup>,
      <a href="https://scholar.google.com/citations?user=ZNE_22wAAAAJ&hl=en" target="_blank">Junyu Luo</a><sup>1</sup>,
      <a href="" target="_blank">Zhongshi Xing</a><sup>3</sup>,
      <a href=""" target="_blank">Hanchun Lian</a><sup>1</sup>,
      <a href="https://scholar.google.com/citations?user=YHbWSOMAAAAJ&hl=zh-CN" target="_blank">Jinsheng Huang</a><sup>1, 2</sup>,
      <a href="https://scholar.google.com/citations?hl=zh-CN&user=YJMJT0gAAAAJ&view_op=list_works&sortby=pubdate" target="_blank">Binqi Chen</a><sup>1,2</sup>,
      <a href="" target="_blank">Luchen Liu</a><sup>2</sup>,
      <a href="https://scholar.google.com/citations?user=1hnJ3TgAAAAJ&hl=en" target="_blank">Yun Ma</a><sup>1&#8224</sup>,
      <a href="https://scholar.google.com/citations?user=LbzoQBsAAAAJ&hl=zh-CN" target="_blank">Ming Zhang</a><sup>1&#8224</sup>,

<div>
  <sup>1</sup>Peking University, <sup>2</sup>Zhengren Quant, <sup>3</sup>Sun Yat-sen University
       </div>   
<div>
<sup>*</sup> Project Leader.
<sup>+</sup> Corresponding author. 
   </div>

</div>

[![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

</div>

This repository contains the official implementation of **MEME** (**M**odeling the **e**volution **m**odes of financial mark**e**ts), a logic-oriented perspective modeling
the financial market as a dynamic, evolutionary ecosystem of competing investment narratives, termed **Modes of Thought** in quantitative finance.

## 🎯 Overview

![MEME Overview](fig/MEME_plot.png)

Current LLM-based financial trading methods are primarily objected-oriented, i.e, focusing on **what to buy and how much should buy**.  We therefore want to investigate the underlying **'why'**  -- capturing the evolving investment logics of financial markets and using this to guide trading.

However, the path to capturing the logics is challenging:

1. Low signal-to-noise ratio of financial markets.

2. The sparse, contradictory nature of  existing narratives in the financial markets.

3. The non-stationary nature of financial markets.

   

## 🚀Key Innovations

**MEME** addresses these challenges by:

1. Structural argument extraction pipeline from the financial markets.
2. Mode identification through probabilistic clustering 
3. Modes evolution through temporal alignment.
4. Portfolio construction through systematic mode evaluation.



## 👉Quick Start

### Environment Configuration

```
conda create --name MEME python==3.10.12 -y
conda activate MEME
pip install pdm
pdm install
```


### Dataset

The dataset used in this project is private and cannot be released due to compliance requirements. We apologize for the inconvenience.  You could use these public data sources as an alternative:

1. **Stock Basic Data and Fundamental Data**
   1. [AkShare](https://akshare.akfamily.xyz/)
   2. [Baostock](https://www.baostock.com/)
   3. [Tushare](https://tushare.pro/)
   4. [Alpha Vantage](https://www.alphavantage.co/)

2. **Financial News and Report**
   1. [East Money](https://data.eastmoney.com/center/)
   2. [Sina](https://vip.stock.finance.sina.com.cn/corp/view/vCB_BulletinGather.php?stock_str=601012&page_index=1)
   3. [Wind](https://www.wind.com.cn/)



### Argument extraction

Configure the pool name (default csi_300), your model url and model api_key in train_pattern.py.

``` python
python train_pattern.py
```

The extracted arguments are stored in `res/analysis/{analyst_type}/{pool_name}` in default. Below is an example of extracted arguments for stock 000001.SZ:

```
{
  "stock_code": "000001",
  "analysis_date": 20231023,
  "analyst_type": "FundamentalAnalyst",
  "analysis_content": [
    {
      "thesis_polar": true,
      "thesis_content": "Ping An Bank's valuation is at an extremely low historical level, with potential for mean reversion. The stock's current P/E ratio of 4.17 and P/B ratio of 0.45 are both at the 0th percentile of historical levels, significantly below the historical medians of 8.84 and 0.82, respectively. Moreover, its return on equity (ROE) has remained stable between 9.9% and 10.8% for five consecutive quarters, indicating robust profitability. This creates a mismatch between the low valuation and sustained earnings."
    },
    {
      "thesis_polar": false,
      "thesis_content": "Continued deterioration in industry fundamentals may negatively impact individual stock performance. The Joint-stock Banks II industry has seen its trailing-twelve-month (TTM) operating revenue growth rate decline from 0.033 to -0.037 over the last five quarters, and its TTM net profit attributable to shareholders growth rate drop from 0.086 to 0.023. This indicates a weakening growth momentum in the sector. During the same period, Ping An Bank's TTM operating revenue growth rate fell from 0.105 to -0.001, reflecting pressure from industry-wide challenges."
    },
    {
      "thesis_polar": true,
      "thesis_content": "High dividend yield provides downside protection and attracts long-term capital. Ping An Bank's current dividend yield of 2.71% is at the 99.9th historical percentile, close to its historical maximum. It is also significantly higher than the industry's historical median dividend yield of 4.86% (with the current industry yield at 5.69%). This high cash return offers allocation value in a low-interest-rate environment."
    },
    {
      "thesis_polar": false,
      "thesis_content": "Slowing asset expansion may limit future growth potential. Ping An Bank's total assets grew from RMB 5.11 trillion in June 2022 to RMB 5.50 trillion in June 2023, representing an annualized growth rate of approximately 7.6%. However, operating revenue declined in the last two quarters (-0.050 and -0.024), indicating that asset expansion has not effectively translated into revenue growth."
    }
  ]
}
```

### Mode identfication and alignment

Configure your embedding model api key on openrouter in train_pattern_aggregation.py.

``` python
python train_pattern_aggregation.py
```

## ☎️ Contact

Please contact the authors below for queries.

- Taian Guo, taianguo@stu.pku.edu.cn
- Haiyang Shen, hyshen@stu.pku.edu.cn