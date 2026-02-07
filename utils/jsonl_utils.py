import json
import signal
import sys

def load_jsonl(file_path):
    """
    Load a JSONL file from the specified file path.
    
    :param file_path: The path to the JSONL file
    :return: A list of JSON objects
    """
    data = []

    success_num = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
                success_num += 1
            except json.decoder.JSONDecodeError as e:
                continue
    
    return data

import json

def dump_jsonl(data, file_path, mode='w', ensure_ascii=False, batch_size=10000):
    """
    将数据以JSONL格式高效地转储到文件路径。
    该函数通过批量写入来优化性能，将多行数据拼接成一个字符串后一次性写入文件。

    :param data: 一个可迭代的对象，其中包含要保存的JSON对象（例如字典列表）。
    :param file_path: 目标文件路径。
    :param mode: 文件打开模式，'w' 表示写入（覆盖），'a' 表示追加。默认为 'w'。
    :param ensure_ascii: 传递给 json.dumps 的参数。默认为 False，以正确处理非ASCII字符。
    :param batch_size: 在写入文件之前要缓冲的行数。
    """
    lines_buffer = []
    try:
        # 使用指定的模式打开文件
        with open(file_path, mode, encoding='utf-8') as file:
            # 遍历数据
            for entry in data:
                # 将JSON对象序列化并添加到缓冲区
                lines_buffer.append(json.dumps(entry, ensure_ascii=ensure_ascii))

                # 当缓冲区达到设定的批量大小时，执行写入
                if len(lines_buffer) >= batch_size:
                    # 将缓冲区中的所有行用换行符连接，并在末尾追加一个换行符，然后写入文件
                    file.write('\n'.join(lines_buffer) + '\n')
                    lines_buffer = []  # 清空缓冲区

            # 循环结束后，写入缓冲区中剩余的所有行
            if lines_buffer:
                file.write('\n'.join(lines_buffer) + '\n')

    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")
        # 可以根据需要添加更复杂的错误处理逻辑

def dump_jsonl_safe(data, file_path, mode='w', ensure_ascii=False, batch_size=10000):
    """
    安全版本的dump_jsonl：
    - 支持 Ctrl-C (KeyboardInterrupt) 时也会把缓冲区的数据完整写入文件。
    - 保证即使发生异常，也不会丢失未写入的数据。
    """

    lines_buffer = []
    file = None

    try:
        file = open(file_path, mode, encoding='utf-8')

        for entry in data:
            lines_buffer.append(json.dumps(entry, ensure_ascii=ensure_ascii))

            if len(lines_buffer) >= batch_size:
                file.write('\n'.join(lines_buffer) + '\n')
                file.flush()   # 保证写入磁盘
                lines_buffer = []

        # 循环结束后，写入缓冲区剩余内容
        if lines_buffer:
            file.write('\n'.join(lines_buffer) + '\n')
            file.flush()

    except KeyboardInterrupt:
        # 写入剩余缓冲区
        if lines_buffer and file:
            try:
                file.write('\n'.join(lines_buffer) + '\n')
                file.flush()
                print(f"\n[警告] 捕获到 Ctrl-C，中断前已保存缓冲区到 {file_path}")
            except Exception as e:
                print(f"\n[错误] 保存缓冲区失败: {e}")

        # 重新抛出，让主程序知道被中断
        raise

    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")
        if lines_buffer and file:
            try:
                file.write('\n'.join(lines_buffer) + '\n')
                file.flush()
                print(f"[警告] 异常发生，但已保存缓冲区数据到 {file_path}")
            except Exception as write_err:
                print(f"[错误] 保存缓冲区失败: {write_err}")
        raise

    finally:
        if file:
            file.close()
