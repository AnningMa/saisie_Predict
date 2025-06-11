class TrieNode:
    
    def __init__(self):

        #if we use [None for _ in range(26)], we can not store the french letters like é,à
        self.children = {}

        #whether this node marks the end of a complete word
        self.isEndOfWord = False

        # number of strings that are stored in the Trie
        #from root node to any Trie node.
        self.wordCount = 0

        #under this node(prefix),frequence of the words.
        #for exemple.:{"apple": 3, "application": 1}
        self.freq_dict = {}