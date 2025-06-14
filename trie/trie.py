from .trie_node import TrieNode

class Trie:

  def __init__(self):
        self.root = TrieNode()

  def insert_key(self, key):

    # Initialize the currentNode pointer with the root node
    currentNode = self.root

    # Iterate across the length of the string
    for c in key:
        # If current character is not in children dict, add a new node
        if c not in currentNode.children:
            currentNode.children[c] = TrieNode()

        # Move to the child node
        currentNode = currentNode.children[c]

        if key in currentNode.freq_dict:
                 currentNode.freq_dict[key] += 1
        else:
                currentNode.freq_dict[key] = 1

        # Update the number of words passing through this node
        currentNode.wordCount += 1

    # Mark the end of a complete word
    currentNode.isEndOfWord = True

  def is_prefix_exist(self, prefix):
    current_node = self.root
    for c in prefix:
      if c not in current_node.children:
        return False

      current_node = current_node.children[c]

    return True

  #check whether a complet word exsits in the Trie
  def search_key(self,key):
    current_node = self.root

    for c in key:
      if c not in current_node.children:
        return False

      current_node = current_node.children[c]

    return current_node.isEndOfWord

  def predict(self, prefix, max_suggestions=5):
    # 首先检查前缀是否存在
    if not self.is_prefix_exist(prefix):
        return []
    
    # 找到前缀对应的节点
    current_node = self.root
    for c in prefix:
        current_node = current_node.children[c]
    
    # 从当前节点收集所有可能的单词及其频率
    suggestions = []
    
    # 如果当前节点本身就是一个完整单词，也包含在建议中
    if current_node.isEndOfWord and prefix in current_node.freq_dict:
        suggestions.append((prefix, current_node.freq_dict[prefix]))
    
    # 收集当前节点的所有单词建议
    for word, freq in current_node.freq_dict.items():
        if word != prefix:  # 避免重复添加前缀本身
            suggestions.append((word, freq))
    
    # 按频率降序排序
    suggestions.sort(key=lambda x: x[1], reverse=True)
    
    # 返回指定数量的建议，只返回单词部分
    return [word for word, freq in suggestions[:max_suggestions]]
