import pandas as pd

# 读取 .ods 文件
df = pd.read_excel("C:\\Users\\Agnes\\Desktop\\talproject\\88milSMS_88522.ods", engine="odf")

# 显示前几行，确认列位置
print(df.head())

# 选择目标列（假设第5列包含短信文本）
sentences = df.iloc[1:, 4].dropna().astype(str).tolist()

# 打印前5个短信和总数量
print(sentences[:5])
print(f"Total: {len(sentences)} sentences")

#traitmennt les phrases
import re
import string

pattern = re.compile(r"<[A-Z]{3}_[0-9]+>")
keep = "'"
punct_to_remove = ''.join(c for c in string.punctuation if c != keep)
translator = str.maketrans(punct_to_remove, ' ' * len(punct_to_remove))

cleaned_sentences = []
for sentence in sentences:
    sentence = str(sentence)
    sentence = pattern.sub("", sentence)  
    sentence = sentence.translate(translator)  

    sentence = re.sub(r'([a-z])([A-Z])', r'\1 \2', sentence)  
    cleaned_sentences.append(sentence)



print(cleaned_sentences[:5])

import random

data_list = cleaned_sentences  # 你的数据列表

random.shuffle(data_list)  # 打乱顺序

n = len(data_list)
split_index = int(n * 0.9)

train_data = data_list[:split_index]
test_data = data_list[split_index:]

print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")