import pandas as pd
import pdb
import copy
import re
import json
from typing import List

def standardize_header_name(name: str) -> str:
    """
    标准化列属性名称。
    规则：转为小写，用下划线替换空格和连字符，移除所有其他非字母数字下划线字符，并合并下划线。
    例如: "Pre-installed OS" -> "pre_installed_os", "Rank #" -> "rank", "Sales (in %)" -> "sales_in"
    """
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = re.sub(r'[\s-]+', '_', s) # 替换空格和连字符为下划线
    s = re.sub(r'[^a-z0-9_]', '', s) # 仅保留字母、数字和下划线
    s = re.sub(r'__+', '_', s) # 将多个下划线合并为单个
    s = s.strip('_') # 移除开头或结尾的下划线
    return s

def extract_markdown_header(md_table_string: str) -> List[str]:
    """
    从一个可能包含Markdown表格的字符串中稳健地提取表头。

    这个函数被设计用来处理各种情况，包括:
    - 表格被 ```markdown ... ``` 代码块包裹。
    - 输入字符串前后有多余的空白或换行符。
    - 表格行的前后有多余的管道符 `|`。
    - 单元格名称（header name）周围有多余的空格。
    - 输入的字符串根本不是一个表格。

    如果成功提取，它会返回一个字符串列表。如果无法提取或输入无效，
    它会返回一个空列表 `[]`。

    :param md_table_string: 包含Markdown表格的字符串。
    :return: 一个包含表头各列名称的列表，或者在失败时返回一个空列表。

    >>> # 标准格式
    >>> extract_markdown_header("| Col A | Col B |\\n|---|---|")
    ['Col A', 'Col B']

    >>> # 被代码块包裹
    >>> extract_markdown_header("```markdown\\n| Name | Age |\\n|---|---|\\n```")
    ['Name', 'Age']

    >>> # 无分隔符行
    >>> extract_markdown_header("| Header 1 | Header 2 |\\n| val 1 | val 2 |")
    ['Header 1', 'Header 2']
    
    >>> # 格式不规范 (多余的空格和管道符)
    >>> extract_markdown_header("  || col1 |  col2   ||  \\n")
    ['', 'col1', 'col2', '']

    >>> # 无效输入：普通文本
    >>> extract_markdown_header("This is just some text.")
    []

    >>> # 无效输入：空字符串或None
    >>> extract_markdown_header("")
    []
    >>> extract_markdown_header(None)
    []
    
    >>> # 边角情况：只有分隔符
    >>> extract_markdown_header("|---|---|")
    ['---', '---']
    """
    # 1. 基本的输入验证和清理
    if not isinstance(md_table_string, str):
        return []
    
    cleaned_string = md_table_string.strip()
    if not cleaned_string:
        return []

    # 2. 处理可选的 ```markdown 代码块
    if cleaned_string.startswith("```markdown"):
        # 稳健地提取代码块内容，防止索引错误
        match = re.search(r"```markdown\s*\n(.*)\n\s*```", cleaned_string, re.DOTALL)
        if match:
            cleaned_string = match.group(1).strip()
        else:
            # 如果格式不匹配（例如没有结尾的```），则认为内容无效
            return []

    # 3. 找到第一个非空行，这应该是表头
    lines = cleaned_string.splitlines()
    header_line = ""
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            header_line = stripped_line
            break
    
    if not header_line:
        return []

    # 4. 验证该行是否可能是表头的一部分（必须包含管道符'|'）
    # 这是一个简单但有效的启发式方法，用于排除普通文本行。
    if '|' not in header_line:
        return []

    # 5. 解析表头行
    # - 首先去除行首和行尾可能存在的多余管道符
    # - 然后按管道符分割
    # - 最后清理每个单元格的空白
    try:
        # .strip('|') 去除开头和结尾的'|'，例如 "| A | B |" -> " A | B "
        columns_raw = header_line.strip('|').split('|')
        
        # 使用列表推导式清理每个表头单元格的空白
        headers = [col.strip() for col in columns_raw]
        
        return headers
    except Exception:
        # 捕获任何意外的解析错误，并返回空列表
        return []


def listdict_to_markdown(data: list, fields: list = None):
    """
    将 list of dict 转成 markdown table（字段名大小写不敏感）
    :param data: list of dict
    :param fields: 要输出的列名（按此顺序），如果 None，则自动收集所有字段（不区分大小写）
    :return: markdown table string
    """
    if not data and fields is None:
        raise Exception("data and fields are both empty")
    
    def normalize_field_name(name):
        """标准化字段名：去掉空格、下划线，转小写"""
        return re.sub(r'[\s_-]+', '', str(name).strip().lower())
    
    # 建立字段名大小写映射: 标准化字段名 -> 原始字段名
    normalized_key_map = {}
    for row in data:
        for key in row.keys():
            normalized_key = normalize_field_name(key)
            if normalized_key not in normalized_key_map:  # 保留第一次出现的原始大小写
                normalized_key_map[normalized_key] = key

    if fields is None:
        # 自动收集所有字段，保持原始字段名
        fields = [normalized_key_map[k] for k in normalized_key_map]
    else:
        # 对 fields 做标准化处理，并映射回原始字段名
        fields = [normalized_key_map.get(normalize_field_name(f), f) for f in fields]

    # 计算每列最大宽度
    col_widths = {f: len(str(f)) for f in fields}
    for row in data:
        for f in fields:
            val = row.get(f, "unknown")
            if val is None:
                val = "unknown"
            col_widths[f] = max(col_widths[f], len(str(val)))

    # 表头
    header_row = "| " + " | ".join(f for f in fields) + " |"

    # 分隔行
    sep_row = "| " + " | ".join("-" * col_widths[f] for f in fields) + " |"

    # 数据行
    data_rows = []
    for row in data:
        cols = []
        is_unknown_row = True
        for f in fields:
            val = row.get(f, "unknown")
            if val is None:
                val = "unknown"
            cols.append(f"{str(val):<{col_widths[f]}}")
            if str(val).strip().lower() not in ["unknown", "-", "na"]:
                is_unknown_row = False
        if not is_unknown_row:  # 过滤掉全是 "unknown" 的行
            line = "| " + " | ".join(cols) + " |"
            data_rows.append(line)

    return "\n".join([header_row, sep_row] + data_rows)


def extract_fields_from_markdown_table(md_table: str, fields: list):
    """
    从markdown表格中提取指定字段，返回只包含这些字段的子表
    :param md_table: markdown表格字符串
    :param fields: 要提取的字段列表（大小写不敏感）
    :return: 只包含指定字段的markdown表格字符串
    """
    if not md_table.strip() or not fields:
        return ""
    
    def normalize_field_name(name):
        """标准化字段名：去掉空格、下划线，转小写"""
        return re.sub(r'[\s_-]+', '', str(name).strip().lower())
    
    # 将markdown表格转换为list of dict
    data = markdown_table_to_listdict(md_table)
    
    if not data:
        return ""
    
    # 建立字段名大小写映射: 标准化字段名 -> 原始字段名
    normalized_key_map = {}
    for row in data:
        for key in row.keys():
            normalized_key = normalize_field_name(key)
            if normalized_key not in normalized_key_map:
                normalized_key_map[normalized_key] = key
    
    # 标准化输入的字段名并映射回原始字段名
    target_fields = []
    for field in fields:
        normalized_field = normalize_field_name(field)
        if normalized_field in normalized_key_map:
            target_fields.append(normalized_key_map[normalized_field])
    
    # 如果没有匹配的字段，返回空表格
    if not target_fields:
        return ""
    
    # 使用现有的listdict_to_markdown函数生成子表格
    return listdict_to_markdown(data, fields=target_fields)

def markdown_table_to_listdict(md_table: str) -> list[dict]:
    """
    将 markdown 表格（字符串）转换为由字典组成的列表。
    此函数对包含或不包含分隔符行的 Markdown 表格都具有鲁棒性。
    它还能处理数据行列数与表头不匹配的情况。

    例如，以下两种格式都能正确解析:
    
    格式1 (带分隔符):
    | Header 1 | Header 2 |
    |----------|----------|
    | val 1    | val 2    |

    格式2 (无分隔符):
    | Header 1 | Header 2 |
    | val 1    | val 2    |

    :param md_table: Markdown 表格字符串
    :return: list[dict] 字典列表
    """
    # 1. 清理输入，去除可选的 ```markdown 代码块
    if md_table.strip().startswith("```markdown"):
        md_table = md_table.strip()[11:-3].strip()
    
    lines = [line.strip() for line in md_table.strip().splitlines() if line.strip()]
    
    # 至少需要表头和一行数据
    if len(lines) < 2:
        return []
    
    # 2. 解析表头
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    num_headers = len(headers)
    
    # 3. 智能检测分隔符行并确定数据行的起始位置
    data_start_index = 1  # 默认假设第二行就是数据
    # 检查第二行是否像分隔符（主要由 '-', '|', ':', ' ' 组成）
    if len(lines) > 1:
        # 清理第二行，只保留非空白、非管道符、非冒号的字符
        second_line_content = re.sub(r'[\s|:]', '', lines[1])
        # 如果清理后只剩下'-'，并且至少有一个'-'，那么它就是分隔符行
        if second_line_content and all(c == '-' for c in second_line_content):
            data_start_index = 2  # 数据从第三行开始
            
    data_lines = lines[data_start_index:]
    
    # 4. 解析数据行
    results = []
    for line in data_lines:
        cols = [c.strip() for c in line.strip("|").split("|")]
        
        # 鲁棒性处理：确保列数与表头数匹配
        if len(cols) > num_headers:
            # 如果列数多于表头数，截断多余的列
            cols = cols[:num_headers]
        elif len(cols) < num_headers:
            # 如果列数少于表头数，用空字符串填充缺失的列
            cols.extend([''] * (num_headers - len(cols)))
            
        row_dict = dict(zip(headers, cols))
        results.append(row_dict)
    
    return results

def update_table_by_keys(current_table, update_table, keys):
    """
    用 update_table 数据更新 current_table（字段名大小写不敏感）。
    如果 update_table 中某行所有字段值都是 "unknown"（大小写不敏感），则跳过新增和更新。
    如果 update_table 中某行部分字段有数据，部分字段没有数据，则仅更新有数据的字段，其他字段保留 current_table 的数据。
    
    :param current_table: list[dict]，现有数据
    :param update_table: list[dict]，更新数据
    :param keys: list[str]，主键字段（大小写不敏感）
    :return: list[dict]，更新后的 current_table
    """
    if not keys:
        raise ValueError("keys 参数不能为空")
    if len(current_table) == 0:
        return copy.deepcopy(update_table)

    def normalize_value(value):
        """对字段值进行标准化处理"""
        return re.sub(r'[\s_-]+', '', str(value).strip().lower())

    def normalize_row(row):
        return {normalize_value(k): normalize_value(v) for k, v in row.items()}

    def make_key(row_dict):
        return tuple(row_dict[k] for k in keys_lower)

    def is_all_unknown(row_dict):
        """判断该行是否所有字段都是 unknown（大小写、空格、下划线不敏感）"""
        return all((normalize_value(v) in ["unknown", '-', 'na']) for v in row_dict.values())
    
    # Normalize keys
    keys_lower = [normalize_value(k) for k in keys]

    current_table_norm = [normalize_row(row) for row in current_table]
    update_table_norm = [normalize_row(row) for row in update_table]

    current_index = {make_key(normalize_row(row)): row for row in current_table}

    for upd_row, upd_row_original in zip(update_table_norm, update_table):
        if is_all_unknown(upd_row):
            # 跳过全 unknown 的更新行
            continue

        pk = make_key(upd_row)
        if pk in current_index:
            # 更新现有行，仅更新有数据的字段
            current_row = current_index[pk]
            for k, v in upd_row_original.items():
                if normalize_value(v) not in ["unknown", '-', 'na']:
                    current_row[k] = v
        else:
            # 添加新行
            current_index[pk] = upd_row_original.copy()

    return list(current_index.values())


def extract_rational_evidence_summary(webpage_content, goal):
    extractor_prompt = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than 10 paragraphs.
2. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "evidence", "summary" feilds**
"""
    return webpage_content
    # prompt = extractor_prompt.format(webpage_content=webpage_content, goal=goal)
    # raw = call_openai(prompt)
    # return raw



def extract_information_with_goal(webpage_content, goal, question, table=None, fields=None, main_fields=None, oss_url=''):
    """
    从文本中提取结构化信息，返回 markdown 表格字符串。
    :param text: 输入文本
    :param goal: 提取目标描述
    :param table: 可选的现有表格数据（markdown 格式字符串）
    :return: markdown 表格字符串
    """

    extractor_prompt = """You are given a user goal, a user question, and a webpage content. Your task is to extract structured information from the webpage content to construct a markdown table that best serves the user's question, using the goal as a sub-goal for this step.

- Consider BOTH the question and the goal: the table should be constructed to answer the question, and the goal describes the current extraction sub-goal.
- Only extract information that is directly relevant and fully certain for the question and goal.
- Do NOT infer, approximate, or paraphrase values—only use information that is explicitly and unambiguously present in the webpage content.
- If a field value is not found or is uncertain, leave it blank or use "-" (do NOT guess or fill with NA).
- Fill in the information truthfully according to the webpage content strictly. Do not make up, speculate or complete the information yourself.
- The table columns should be determined by the fields and the information required to fulfill the question and goal.

**Output Format:**
A markdown table containing all the extracted rows (not incremental, but a new table) in the tags <Table>...</Table>.

**Important:** 
1. The table should only contain information that is exactly and unambiguously found in the webpage content.
2. Do NOT include extra or irrelevant information.
3. Pay careful attention to the units and details required by the question and goal.
4. If no relevant information is found, output an empty markdown table with only the header.

Example:

fields: ["Date", "Concert’s English Name"]

Goal:
Extract all concerts with their date, English name, host country, host city, and host venue.

Question:
What are the dates and English names of all concerts, along with their host country, city, and venue?

Webpage Content:
In 6th February, 2010, Australia, the concert held in Sydney.

Output:
<Table>
| Date               | Concert’s English Name  | Host Country | Host City | Host Venue                    | 
|------------------- | ----------------------- | ------------ | --------- | ----------------------------- |
| 6th February, 2010 | -                       | Australia    | Sydney    | -                             | 
</Table>

Now begin,

## **Goal**
{goal}

## **Question**
{question}

## **fields**
{fields}

## **Webpage Content**
{webpage_content}
""" 

    text = extract_rational_evidence_summary(webpage_content, goal)
    prompt = extractor_prompt.format(webpage_content=text, goal=goal, question=question, fields=json.dumps(fields))
    # def post_fn(text):
    #     table_ = re.findall(r'<Table>(.*?)</Table>', text, re.DOTALL)[-1].strip()
    #     return table_   
    # table_ = call_openai(prompt, post_fn)
    def post_fn(text):

        text = text.split('<|start|>assistant<|channel|>final<|message|>')[-1].split('<|return|>')[0].strip()
        table_ = re.findall(r'<Table>(.*?)</Table>', text, re.DOTALL)[-1].strip()
        return table_ 

    # table_ = call_llm(prompt, system_prompt='You are a helpful assistant.', TOOLS_SCHEMA=None, first=True, reasoning_effort="medium", post_fn=post_fn, oss_url=oss_url)[0]
    table_ = call_llm(prompt, system_prompt='You are a helpful assistant.', TOOLS_SCHEMA=None, first=True, reasoning_effort="low", post_fn=post_fn, oss_url=oss_url)[0]

    
    
    return table_


def extract_information(webpage_content, question, table=None, fields=None, main_fields=None, oss_url=''):
    """
    从文本中提取结构化信息，返回 markdown 表格字符串。
    :param webpage_content: 输入文本
    :param question: 提取目标描述
    :param table: 可选的现有表格数据（markdown 格式字符串）
    :return: markdown 表格字符串
    """

    extractor_prompt = """You are given a user question and a webpage content. Your task is to extract structured information from the webpage content to construct a markdown table that best serves the user's question.

- Only extract information that is directly relevant and fully certain for the question.
- Do NOT infer, approximate, or paraphrase values—only use information that is explicitly and unambiguously present in the webpage content.
- If a field value is not found or is uncertain, leave it blank or use "-" (do NOT guess or fill with NA).
- Fill in the information truthfully according to the webpage content strictly. Do not make up, speculate or complete the information yourself.
- The table columns should be determined by the fields and the information required to fulfill the question.

**Output Format:**
A markdown table containing all the extracted rows (not incremental, but a new table) in the tags <Table>...</Table>.

**Important:** 
1. The table should only contain information that is exactly and unambiguously found in the webpage content.
2. Do NOT include extra or irrelevant information.
3. Pay careful attention to the units and details required by the question.
4. If no relevant information is found, output an empty markdown table with only the header.

Example:

fields: ["Date", "Concert’s English Name"]

Question:
What are the dates and English names of all concerts, along with their host country, city, and venue?

Webpage Content:
In 6th February, 2010, Australia, the concert held in Sydney.

Output:
<Table>
| Date               | Concert’s English Name  | Host Country | Host City | Host Venue                    | 
|------------------- | ----------------------- | ------------ | --------- | ----------------------------- |
| 6th February, 2010 | -                       | Australia    | Sydney    | -                             | 
</Table>

Now begin,

## **Question**
{question}

## **fields**
{fields}

## **Webpage Content**
{webpage_content}
""" 

    # 这里不再调用 extract_rational_evidence_summary
    prompt = extractor_prompt.format(webpage_content=webpage_content, question=question, fields=json.dumps(fields))

    # def post_fn(text):
    #     table_ = re.findall(r'<Table>(.*?)</Table>', text, re.DOTALL)[-1].strip()
    #     return table_ 
    # table_ = call_openai(prompt, post_fn)

    def post_fn(text):

        text = text.split('<|start|>assistant<|channel|>final<|message|>')[-1].split('<|return|>')[0].strip()
        table_ = re.findall(r'<Table>(.*?)</Table>', text, re.DOTALL)[-1].strip()
        return table_ 

    # table_ = call_llm(prompt, system_prompt='You are a helpful assistant.', TOOLS_SCHEMA=None, first=True, reasoning_effort="medium", post_fn=post_fn, oss_url=oss_url)[0]
    table_ = call_llm(prompt, system_prompt='You are a helpful assistant.', TOOLS_SCHEMA=None, first=True, reasoning_effort="low", post_fn=post_fn, oss_url=oss_url)[0]

    return table_






if __name__ == "__main__":
#     # 示例数据
#     table1 = '''| fiscal year | federal budget | federal spending | federal deficit | national debt | net interest cost |
# | ----------- | -------------- | ---------------- | --------------- | ------------- | ----------------- |'''
#     table2 = '''| fiscal year | federal budget | federal spending | federal deficit | national debt | net interest cost |
# | ----------- | -------------- | ---------------- | --------------- | ------------- | ----------------- |
# | FY2015     | -              | 3.688           | 0.439          | -             | -                 |'''
#     table3 = '''| fiscal year | federal budget | federal spending | federal deficit | national debt | net interest cost |
# | ----------- | -------------- | ---------------- | --------------- | ------------- | ----------------- |
# | FY2015      | -              | 3.688            | 0.438           | -             | -                 |
# | FY2016      | -              | 3.853            | 0.585           | -             | -                 |
# | FY2017      | -              | 3.982            | 0.665           | -             | -                 |
# | FY2018      | -              | 4.173            | 0.833           | -             | -                 |'''
#     table4 = '''| fiscal year | federal budget | federal spending | federal deficit | national debt | net interest cost |
# | ----------- | -------------- | ---------------- | --------------- | ------------- | ----------------- |'''

#     table = update_table_by_keys(markdown_table_to_listdict(table1), markdown_table_to_listdict(table2), keys=["fiscal year"])
#     print(listdict_to_markdown(table))
#     print()
#     table = update_table_by_keys(table, markdown_table_to_listdict(table3), keys=["fiscal year"])
#     print(listdict_to_markdown(table))
#     print()
#     table = update_table_by_keys(table, markdown_table_to_listdict(table4), keys=["fiscal year"])
#     print(listdict_to_markdown(table))
#     print()



        # 示例数据
    table1 = '''| fiscal year | federal budget | federal spending | federal deficit | national debt | net interest cost |
| ----------- | -------------- | ---------------- | --------------- | ------------- | ----------------- |
| FY2015      | -              | 3.688            | -0.438          | 18.151        | 0.223             |
| FY2016      | -              | 3.853            | -0.585          | 19.573        | 0.241             |
| FY2017      | -              | 3.982            | -0.665          | 20.245        | 0.263             |
| FY2018      | -              | 4.109            | -0.779          | 21.516        | 0.325             |
| FY2019      | -              | 4.447            | -0.984          | 22.719        | 0.375             |
| FY2020      | -              | 6.554            | -3.132          | 26.945        | 0.346             |
| FY2021      | -              | 6.822            | -2.775          | 28.428        | 0.352             |
| FY2022      | -              | 6.273            | -1.376          | 30.928        | 0.475             |
| FY2023      | -              | 6.135            | -1.694          | 32.658        | 0.640             |
| FY2024      | -              | 6.752            | -1.893          | 34.814        | 0.879             |'''
    table2 = '| fiscal year | federal budget | federal spending | federal deficit | national debt | net interest cost |\n| ----------- | -------------- | ---------------- | --------------- | ------------- | ----------------- |'

    table3 = '''| fiscal year | federal budget | federal spending | federal deficit | national debt | net interest cost |
| ----------- | -------------- | ---------------- | --------------- | ------------- | ----------------- |
| FY2015      | -              | 3.688            | 0.438           | -             | -                 |
| FY2016      | -              | 3.853            | 0.585           | -             | -                 |
| FY2017      | -              | 3.982            | 0.665           | -             | -                 |
| FY2018      | -              | 4.173            | 0.833           | -             | -                 |'''


    table4 = '''| fiscal year | federal budget | federal spending | federal deficit | national debt | net interest cost |
| ----------- | -------------- | ---------------- | --------------- | ------------- | ----------------- |'''

    # table = update_table_by_keys(markdown_table_to_listdict(table1), markdown_table_to_listdict(table2), keys=["Fiscal Year"])
    # print(listdict_to_markdown(table))
    # print()
    # table = update_table_by_keys(table, markdown_table_to_listdict(table3), keys=["fiscal year"])
    # print(listdict_to_markdown(table, fields=["fiscal year", "national debt"]))
    # print()

    # table = update_table_by_keys(table, markdown_table_to_listdict(table4), keys=["fiscal year"])
    # print(listdict_to_markdown(table))
    # print()

    webpage_content = '''Historical development
The following table shows the development of annual expenditure and revenue of the United States federal government.[130]

Year	Revenues
(million $)	Outlays
(million $)	Deficit
(million $)	Deficit
(in % of GDP)
1789–1849 (total)	1,160	1,090	70	
1850–1900 (total)	14,462	15,453	−991	
1901	588	525	63	
1902	562	485	77	
1903	562	517	45	
1904	541	584	−43	
1905	544	567	−23	
1906	595	570	25	
1907	666	579	87	
1908	602	659	−57	
1909	604	694	−89	
1910	676	694	−18	
1911	702	691	11	
1912	693	690	3	
1913	714	715	−1	
1914	725	726	−1	
1915	683	746	−63	
1916	761	713	48	
1917	1,101	1,954	−853	
1918	3,645	12,677	−9,032	
1919	5,130	18,493	−13,363	
1920	6,649	6,358	291	
1921	5,571	5,062	509	
1922	4,026	3,289	736	
1923	3,853	3,140	713	
1924	3,871	2,908	963	
1925	3,641	2,924	717	
1926	3,795	2,930	865	
1927	4,013	2,857	1,155	
1928	3,900	2,961	939	
1929	3,862	3,127	734	
1930	4,058	3,320	738	0.8%
1931	3,116	3,577	−462	−0.5%
1932	1,924	4,659	−2,735	−4.0%
1933	1,997	4,598	−2,602	−4.5%
1934	2,955	6,541	−3,586	−5.8%
1935	3,609	6,412	−2,803	−4.0%
1936	3,923	8,228	−4,304	−5.4%
1937	5,387	7,580	−2,193	−2.5%
1938	6,751	6,840	−89	−0.1%
1939	6,295	9,141	−2,846	−3.1%
1940	6,548	9,468	−2,920	−3.0%
1941	8,712	13,653	−4,941	−4.3%
1942	14,634	35,137	−20,503	−13.9%
1943	24,001	78,555	−54,554	−29.6%
1944	43,747	91,304	−47,557	−22.2%
1945	45,159	92,712	−47,553	−21.0%
1946	39,296	55,232	−15,936	−7.0%
1947	38,514	34,496	4,018	1.7%
1948	41,560	29,764	11,796	4.5%
1949	39,415	38,835	580	0.2%
1950	39,443	42,562	−3,119	−1.1%
1951	51,616	45,514	6,102	1.9%
1952	66,167	67,686	−1,519	−0.4%
1953	69,608	76,101	−6,493	−1.7%
1954	69,701	70,855	−1,154	−0.3%
1955	65,451	68,444	−2,993	−0.7%
1956	74,587	70,640	3,947	0.9%
1957	79,990	76,578	3,412	0.7%
1958	79,636	82,405	−2,769	−0.6%
1959	79,249	92,098	−12,849	−2.5%
1960	92,492	92,191	301	0.1%
1961	94,388	97,723	−3,335	−0.6%
1962	99,676	106,821	−7,146	−1.2%
1963	106,560	111,316	−4,756	−0.8%
1964	112,613	118,528	−5,915	−0.9%
1965	116,817	118,228	−1,411	−0.2%
1966	130,835	134,532	−3,698	−0.5%
1967	148,822	157,464	−8,643	−1.0%
1968	152,973	178,134	−25,161	−2.8%
1969	186,882	183,640	3,242	0.3%
1970	192,807	195,649	−2,842	−0.3%
1971	187,139	210,172	−23,033	−2.1%
1972	207,309	230,681	−23,373	−1.9%
1973	230,799	245,707	−14,908	−1.1%
1974	263,224	269,359	−6,135	−0.4%
1975	279,090	332,332	−53,242	−3.3%
1976	298,060	371,792	−73,732	−3.1%
1977	355,559	409,218	−53,659	−2.6%
1978	399,561	458,746	−59,185	−2.6%
1979	463,302	504,028	−40,726	−1.6%
1980	517,112	590,941	−73,830	−2.6%
1981	599,272	678,241	−78,968	−2.5%
1982	617,766	745,743	−127,977	−3.9%
1983	600,562	808,364	−207,802	−5.9%
1984	666,438	851,805	−185,367	−4.7%
1985	734,037	946,344	−212,308	−5.0%
1986	769,155	990,382	−221,227	−4.9%
1987	854,287	1,004,017	−149,730	−3.1%
1988	909,238	1,064,416	−155,178	−3.0%
1989	991,104	1,143,743	−152,639	−2.7%
1990	1,031,958	1,252,993	−221,036	−3.7%
1991	1,054,988	1,324,226	−269,238	−4.4%
1992	1,091,208	1,381,529	−290,321	−4.5%
1993	1,154,334	1,409,386	−255,051	−3.8%
1994	1,258,566	1,461,752	−203,186	−2.8%
1995	1,351,790	1,515,742	−163,952	−2.2%
1996	1,453,053	1,560,484	−107,431	−1.3%
1997	1,579,232	1,601,116	−21,884	−0.3%
1998	1,721,728	1,652,458	69,270	0.8%
1999	1,827,452	1,701,842	125,610	1.3%
2000	2,025,191	1,788,950	236,241	2.3%
2001	1,991,082	1,862,846	128,236	1.2%
2002	1,853,136	2,010,894	−157,758	−1.5%
2003	1,782,314	2,159,899	−377,585	−3.3%
2004	1,880,114	2,292,841	−412,727	−3.4%
2005	2,153,611	2,471,957	−318,346	−2.5%
2006	2,406,869	2,655,050	−248,181	−1.8%
2007	2,567,985	2,728,686	−160,701	−1.1%
2008	2,523,991	2,982,544	−458,553	−3.1%
2009	2,104,989	3,517,677	−1,412,688	−9.8%
2010	2,162,706	3,457,079	−1,294,373	−8.7%
2011	2,303,466	3,603,065	−1,299,599	−8.5%
2012	2,449,990	3,536,945	−1,086,955	−6.8%
2013	2,775,106	3,454,648	−679,542	−4.1%
2014	3,021,491	3,506,091	−484,600	−2.8%
2015	3,249,887	3,688,383	−438,496	−2.4%
2016	3,267,961	3,852,612	−584,651	−3.3%
2017	3,316,182	3,981,554	−665,372	−3.7%
2018	3,329,907	4,109,047	−779,140	−3.9%
2019	3,463,364	4,446,960	−983,596	−4.7%
2020	3,421,164	6,553,621	−3,132,457	−14.9%
2021	4,047,111	6,822,470	−2,775,359	−12.0%
2022	4,897,399	6,273,324	−1,375,925	−5.4%
2023	4,440,947	6,134,672	−1,693,725	−6.3%
2024	4,919,000	6,752,000	−1,893,000	−6.4%
'''

    goal = "Extract all fiscal years with their federal budget, federal spending, federal deficit, national debt, and net interest cost."
    question = "Need to analyze the trend of the U.S. federal government spending and deficit before and after the pandemic. Please provide me with the following data through fiscal years 2015-2024: the federal Budget (trillion), the federal spending (trillion), the federal deficit (trillion), the national debt (trillion), and the net interest cost on the gross federal debt (trillion).\n\nPlease organize the results in one Markdown table with the following column names in order: \nFiscal Year, Federal Budget, Federal Spending, Federal Deficit, National Debt, Net Interest Cost.\n \nUnder the Fiscal Year, state the statistics like FY2015, FY2016.\n\nDon't ask me any questions, just output the results according to the columns without omitting cells arbitrarily. "
    main_fields = ["fiscal year"]
    fileds = ["fiscal year", "federal budget", "federal spending", "federal deficit", "national debt", "net interest cost"]
    
    out = extract_information(webpage_content, question, table=None, fields=fileds, main_fields=main_fields)
    print(out)
    out = extract_information_with_goal(webpage_content, goal, question, table=None, fields=fileds, main_fields=main_fields)
    print(out)