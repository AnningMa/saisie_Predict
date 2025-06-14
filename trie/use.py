import data
from .trie import Trie

# 初始化 Trie
My_trie = Trie()

def build_trie_from_data():
    
    for sentence in data.train_data:
        if not sentence or sentence.isspace():
            continue
        words = sentence.split()
        for word in words:
            word = word.strip()
            if word:
                My_trie.insert_key(word.lower())

    print(f"trie est établi ")
    print(f"nombre de phrase {len(data.train_data)}")

def completition():
    
    while True:
        prefix = input("input a prefix:").strip().lower()

        if prefix == 'quit':
            break

        suggestions = My_trie.predict(prefix)
        for word in suggestions:
          print(word,end=" ")
        print()

if __name__ == "__main__":
    build_trie_from_data()
    completition()
