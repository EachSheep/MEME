import json
import re

def check_and_replace(text, pattern, replacement):
    """
    Check if the pattern exists in the text. If it exists, replace it with the replacement text.
    If it doesn't exist, raise an exception.

    Parameters:
    text (str): The text to be processed.
    pattern (str): The pattern to search for.
    replacement (str): The text to replace the pattern with.

    Returns:
    str: The text after replacement.

    Exceptions:
    ValueError: Raised if the pattern is not found in the text.
    """
    if pattern in text:
        return text.replace(pattern, replacement)
    else:
        raise ValueError(f"Pattern '{pattern}' not found in the text.")

def extract_largest_markdown_block(text: str) -> str | None:
    """
    从一个字符串中提取被 ```markdown...``` 包裹的最大内容块。

    Args:
        text (str): 包含 markdown 代码块的原始字符串。

    Returns:
        str | None: 如果找到，则返回最长的内容字符串；如果没有找到任何匹配项，则返回 None。
    """
    # 正则表达式模式：
    # ```markdown\n  - 匹配起始标记
    # (.*?)          - 非贪婪模式匹配任意字符（包括换行符），并将其捕获
    # \n```          - 匹配结束标记
    # re.DOTALL 标志让 . 可以匹配换行符
    pattern = r"```markdown(.*?)```"
    
    # 查找所有非重叠的匹配项
    # re.findall会返回所有捕获组（括号里的内容）的列表
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    # 如果没有找到任何匹配项，返回 None
    if not matches:
        return None
    
    # 使用 max() 函数和 key=len 来找到列表中最长的字符串
    return max(matches, key=len)

def extract_largest_json(text: str):
    """
    从字符串中提取并解析最大的、由 '{}' 包裹的有效JSON对象。

    该函数会查找所有由匹配的 '{}' 包裹的顶级块，然后按长度从大到小排序。
    它会依次尝试解析这些块，并返回第一个成功解析的Python对象（字典）。
    如果所有块都解析失败或字符串中没有 '{}' 块，则返回None。

    Args:
        text (str): 包含JSON数据的原始字符串。

    Returns:
        dict or None: 成功解析出的最大的JSON对象（字典），否则返回None。
    """
    # 用于存储所有可能的顶级JSON字符串的候选列表
    candidates = []
    # 使用栈的思路来匹配括号，brace_level相当于栈的深度
    brace_level = 0
    start_index = -1

    # 遍历字符串，寻找所有由匹配的 '{}' 包裹的顶级块
    for i, char in enumerate(text):
        if char == '{':
            if brace_level == 0:
                # 记录一个顶级块的开始位置
                start_index = i
            brace_level += 1
        elif char == '}':
            if brace_level > 0:
                brace_level -= 1
                if brace_level == 0 and start_index != -1:
                    # 发现一个完整的顶级块，提取并添加到候选列表
                    candidate_str = text[start_index : i + 1]
                    candidates.append(candidate_str)
                    start_index = -1

    # 如果没有找到任何候选块，直接返回None
    if not candidates:
        return None

    # 按长度对候选块进行降序排序，这样我们就能先尝试最长的
    candidates.sort(key=len, reverse=True)

    # 从最长的候选块开始，尝试进行JSON解析
    for candidate in candidates:
        try:
            # json.loads() 用于从字符串解析JSON
            parsed_json = json.loads(candidate)
            # 只要有一个成功了，就立即返回结果，因为它就是最大的有效JSON
            return parsed_json
        except json.JSONDecodeError:
            # 如果解析失败，说明这个块不是有效的JSON，继续尝试下一个
            continue

    # 如果所有候选块都解析失败，返回None
    return None

def find_largest_thought(text: str, pattern: str = "thought") -> str | None:
    """
    从输入字符串中匹配所有由 <thought></thought> 包裹的内容，
    并返回其中最长的一个。

    Args:
        text: 包含<thought>标签的原始字符串。

    Returns:
        最长的匹配内容字符串。如果没有找到任何匹配项，则返回 None。
    """
    # 正则表达式模式：
    # <thought>      - 匹配开头的标签
    # (.*?)          - 非贪婪模式匹配中间的所有字符（包括换行符）
    #                - () 表示一个捕获组，findall只会返回这部分内容
    # </thought>     - 匹配结尾的标签
    # re.DOTALL 标志让 . 可以匹配换行符
    pattern = r"<thought>(.*?)</thought>"
    
    # 使用 re.findall 找到所有匹配项
    # findall会返回一个列表，包含所有捕获组（也就是括号里的内容）
    matches = re.findall(pattern, text, re.DOTALL)
    
    # 如果没有找到任何匹配项，返回 None
    if not matches:
        return None
    
    # 使用 max() 函数和 key=len 找出列表中最长的字符串
    largest_match = max(matches, key=len)
    
    return largest_match