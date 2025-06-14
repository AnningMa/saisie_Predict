import sys
sys.path.append("C:\\Users\\Agnes\\Desktop\\talproject")  
import data 
from .trie import Trie
import random
from .use import My_trie

def test_predict_with_prefix(trie, test_data):

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

            prefix_len = 3
            prefix = word[:prefix_len]

            predictions = trie.predict(prefix, max_suggestions=3)

            if word.lower() in (w.lower() for w in predictions):
                correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Total words tested: {total}")
    print(f"Correctly predicted words: {correct}")
    print(f"Prefix prediction accuracy: {accuracy:.2%}")

# 运行测试
test_predict_with_prefix(My_trie, data.test_data)