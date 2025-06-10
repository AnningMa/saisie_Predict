import data
from trie import Trie

trie = Trie()
word_count = 0
unique_words = set()

for i, sentence in enumerate(data.train_data):
    if not sentence or sentence.isspace():  # vide phrase
        continue
    words = sentence.split()

    for word in words:
      word = word.strip()
      if word:
        trie.insert_key(word.lower())
        unique_words.add(word.lower())
        word_count += 1


print(f"trie est établi ")
print(f"nombre de phrase {len(data.train_data)}")
print(f"nombre de mot: {word_count}")
print(f"nombre de unique mot: {len(unique_words)}")


