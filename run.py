from ngramme.ngramme import *
from trie.trie import *
import data
from trie.use import My_trie, completition, build_trie_from_data

if __name__ == "__main__":
    # print("Complétion")
    # build_trie_from_data()
    # completition()
    
    print("Prédiction")
    model = create_model(data.train_data, 7) 
    interactive_test(model)
    

    
    
