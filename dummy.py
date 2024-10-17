import re

def truncate_string(input_str):
    # 定义匹配 '\n', '.', ',', 'Question' 的正则表达式
    pattern = r'[\n.,]|Question'
    
    # 使用正则表达式进行分割
    result = re.split(pattern, input_str, maxsplit=1)
    
    # 返回分割后的第一个部分（截断后的字符串）
    return result[0]

# 示例字符串
input_str = "This is a test string. This will be truncated at the first period, or a newline\nor a comma, or if we see the word Question."
input_str = 'This is'

# 截断字符串
truncated = truncate_string(input_str)
print(truncated)