import data
from trie import Trie
import random
from use import trie

def test_predict_with_prefix(trie, test_data):
    """
    测试Trie的预测功能 - 针对法语数据
    """
    correct = 0
    total = 0

    for sentence in test_data:
        if not sentence or sentence.isspace():
            continue
            
        words = sentence.split()
        
        for word in words:
            if len(word) < 2:
                continue
                
            total += 1

            # 取单词的1或2个字母作为前缀
            prefix_len = random.choice([2, 3])
            prefix = word[:prefix_len]

            # trie预测以prefix开头的单词列表
            predictions = trie.predict(prefix, max_suggestions=5)

            # 检查预测结果里是否有正确单词（忽略大小写）
            if word.lower() in (w.lower() for w in predictions):
                correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Total words tested: {total}")
    print(f"Correctly predicted words: {correct}")
    print(f"Prefix prediction accuracy: {accuracy:.2%}")

# 运行测试
test_predict_with_prefix(trie, data.test_data)