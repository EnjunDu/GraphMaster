# ./tricks/generate_data_clear.py
import json
import re

def fix_messy_json(content):
    """
    修正混乱的 JSON 文本，返回包含多个字典的列表。
    预处理步骤：
      1. 去除最外层中括号，如果存在的话；
      2. 如果整个内容被包在引号内（即成为一个字符串），去掉外层引号；
      3. 将转义换行符转换成实际换行符；
      4. 利用特定分隔符拆分各个对象，确保每个对象以 { 开始，以 } 结束后解析。
    """
    content = content.strip()
    # 如果内容以 [ 开始且以 ] 结束，则去除最外层中括号
    if content.startswith('[') and content.endswith(']'):
        content = content[1:-1].strip()

    # 如果内容被包裹在引号内，则去掉最外层引号
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1].strip()

    # 将转义的换行符（\n）转换为实际换行符
    content = content.replace(r'\n', '\n')

    # 有时各个对象之间的分隔符为 "},\n  {"，我们用一个特殊分隔符来分割各个对象
    delimiter = "|DELIM|"
    content = content.replace('},\n  {', '}' + delimiter + '{')

    # 分割为各个对象的字符串
    parts = content.split(delimiter)

    objs = []
    for i, part in enumerate(parts):
        part = part.strip()
        # 如果部分不以 { 开始，则补上
        if not part.startswith('{'):
            part = '{' + part
        # 如果部分不以 } 结束，则补上
        if not part.endswith('}'):
            part = part + '}'
        # 去除可能前后多余的逗号
        part = part.strip(', \n')
        try:
            obj = json.loads(part)
            objs.append(obj)
        except Exception as e:
            print(f"解析第 {i} 个对象时出错: {e}")
            print("内容：", part)
    return objs

def reformat_json_file(input_file, output_file):
    """
    读取混乱格式的 JSON 文件，转换为标准格式，并写入输出文件。
    标准格式示例：
    [
        {
            "node_id": 0,
            "label": "2",
            "text": "Title: ... Abstract: ...",
            "neighbors": [8, 14, 258, 435, 544]
        },
        {
            "node_id": 1,
            "label": "5",
            "text": "Title: ... Abstract: ...",
            "neighbors": [344]
        },
        ...
    ]
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 首先尝试直接解析整个文件内容
    try:
        data = json.loads(content)
    except Exception as e:
        print("直接解析 JSON 出错，尝试手动修正。错误信息：", e)
        data = None

    objs = []
    if data is None:
        # 如果直接解析失败，则调用手动修正函数
        objs = fix_messy_json(content)
    else:
        # 如果解析成功，则判断数据结构
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # 已经是标准的列表字典格式，直接使用
                objs = data
            elif len(data) > 0 and isinstance(data[0], str):
                # 如果列表中存储的是字符串，假设为一个混乱的文本，尝试修正它
                messy = data[0]
                objs = fix_messy_json(messy)
            else:
                print("不支持的 JSON 数据结构！")
                return
        else:
            print("不支持的 JSON 数据结构！")
            return



    # 将结果写入输出文件（标准格式，缩进4格，保留中文）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(objs, f, indent=4, ensure_ascii=False)
    print(f"转换完成，标准格式的 JSON 已写入 {output_file}")

if __name__ == '__main__':
    # 请根据实际情况指定输入输出文件路径
    input_file = 'cora_augmented_output (3).json'
    output_file = 'output.json'
    reformat_json_file(input_file, output_file)
