from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import math

class NGramPredictor:
  
    def __init__(self, n: int = 3):
        
        self.n = n
        # 存储n-gramme及其后续词的频率
        self.ngram_counts = defaultdict(Counter)  # {(w1, w2): {next_word: count}}
        self.context_counts = defaultdict(int)    # {(w1, w2): total_count}
        self.vocabulary = set()
    
    def train_from_sentences(self, sentences: List[str]):
      
        print(f"train {self.n}-gramme 模型...")
        
        for sentence in sentences:
            words = self._clean_and_tokenize(sentence)
            if len(words) == 0:
                continue
                
            padded_sentence = ['<START>'] * (self.n - 1) + words + ['<END>']
            
            for i in range(len(padded_sentence) - self.n + 1):
                
                context = tuple(padded_sentence[i:i + self.n - 1])
                next_word = padded_sentence[i + self.n - 1]
                
                self.ngram_counts[context][next_word] += 1
                self.context_counts[context] += 1
                self.vocabulary.add(next_word)
        
        print(f"train finish: {len(self.ngram_counts)} , length: {len(self.vocabulary)}")
    
    def _clean_and_tokenize(self, sentence: str) -> List[str]:
       
        import re
        # remove the space
        sentence = re.sub(r'\s+', ' ', sentence.strip().lower())
        words = sentence.split()
        return [word for word in words if word]  #vide 
    
    def get_word_probability(self, context: Tuple[str, ...], word: str) -> float:
        
        if context not in self.context_counts:
            return 0.0
        
        word_count = self.ngram_counts[context][word]
        total_count = self.context_counts[context]
        
        smoothed_prob = (word_count + 1) / (total_count + len(self.vocabulary))
        return smoothed_prob
    
    def predict_next_words(self, context_words: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
       
        if len(context_words) >= self.n - 1:
            context = tuple(context_words[-(self.n - 1):])
        else:
            padding = ['<START>'] * (self.n - 1 - len(context_words))
            context = tuple(padding + context_words)
        
        predictions = []
        
       
        if context in self.ngram_counts:
            for word, count in self.ngram_counts[context].items():
                if word != '<END>':  
                    prob = self.get_word_probability(context, word)
                    predictions.append((word, prob))
        
       
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]
    
    def predict_with_backoff(self, context_words: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
       
        predictions = []
        
        for n in range(min(self.n, len(context_words) + 1), 0, -1):
            if n == 1:
                
                word_counts = Counter()
                for ngram_dict in self.ngram_counts.values():
                    for word, count in ngram_dict.items():
                        if word != '<END>':
                            word_counts[word] += count
                
                total = sum(word_counts.values())
                for word, count in word_counts.most_common(top_k):
                    predictions.append((word, count / total))
                break
            else:
             
                context = tuple(context_words[-(n-1):]) if len(context_words) >= n-1 else tuple(['<START>'] * (n-1-len(context_words)) + context_words)
                
                if context in self.ngram_counts:
                    for word, count in self.ngram_counts[context].items():
                        if word != '<END>':
                            prob = self.get_word_probability(context, word)
                            predictions.append((word, prob))
                    
                    if predictions:
                        break
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]
    
    def evaluate_model(self, test_sentences: List[str], top_k: int = 5) -> Dict[str, float]:
        
        total_predictions = 0
        correct_predictions = {f'top_{i}': 0 for i in range(1, top_k + 1)}
        total_perplexity = 0.0
        valid_perplexity_count = 0
        
        for sentence in test_sentences:
            words = self._clean_and_tokenize(sentence)
            if len(words) < self.n:  # too short
                continue
            
            # add [start]
            padded_words = ['<START>'] * (self.n - 1) + words
            
            # predict
            for i in range(self.n - 1, len(padded_words)):
                # context
                context = padded_words[i - (self.n - 1):i]
                actual_word = padded_words[i]
                
                predictions = self.predict_next_words(context, top_k)
                
                if predictions:  
                    total_predictions += 1
                    
                    predicted_words = [word for word, _ in predictions]
                    for k in range(1, min(len(predicted_words) + 1, top_k + 1)):
                        if actual_word in predicted_words[:k]:
                            correct_predictions[f'top_{k}'] += 1
                    
                    actual_prob = self.get_word_probability(tuple(context), actual_word)
                    if actual_prob > 0:
                        total_perplexity += math.log(actual_prob)
                        valid_perplexity_count += 1
        
        results = {}
        
        for k in range(1, top_k + 1):
            if total_predictions > 0:
                accuracy = correct_predictions[f'top_{k}'] / total_predictions
                results[f'accuracy_top_{k}'] = accuracy
            else:
                results[f'accuracy_top_{k}'] = 0.0
        
        if valid_perplexity_count > 0:
            avg_log_prob = total_perplexity / valid_perplexity_count
            perplexity = math.exp(-avg_log_prob)
            results['perplexity'] = perplexity
        else:
            results['perplexity'] = float('inf')
        
        results['total_predictions'] = total_predictions
        results['test_coverage'] = total_predictions / max(1, sum(len(self._clean_and_tokenize(s)) for s in test_sentences))
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, float]):
      
        print("accurancy:")
        for key, value in results.items():
            if key.startswith('accuracy_top_'):
                k = key.split('_')[-1]
                print(f"  Top-{k}: {value:.2%}")
        print()
        
        # perplexity
        if results['perplexity'] != float('inf'):
            print(f"perplexity: {results['perplexity']:.2f}")
        else:
            print("cannot have perpexity）")

def test_ngram_model(train_data: List[str], test_data: List[str], n: int = 3):
    
    # creat the model
    model = NGramPredictor(n=n)
    model.train_from_sentences(train_data)
    
    # evaluation
    results = model.evaluate_model(test_data, top_k=5)
    model.print_evaluation_results(results)
    
    return model, results


def compare_different_n_values(train_data: List[str], test_data: List[str], n_values: List[int] = [2, 3, 4]):

    results = {}
    
    for n in n_values:
        print(f"\n{'='*60}")
        model, eval_results = test_ngram_model(train_data, test_data, n)
        results[n] = {
            'model': model,
            'evaluation': eval_results
        }
    

    print("compare the model:")
 
    print(f"{'N-gramme':<10}{'Top-1 Acc':<12}{'Top-3 Acc':<12}{'Top-5 Acc':<12}{'Perplexity':<12}")

    
    for n in n_values:
        eval_data = results[n]['evaluation']
        print(f"{n}-gramme   ", end="")
        print(f"{eval_data['accuracy_top_1']:<12.2%}", end="")
        print(f"{eval_data['accuracy_top_3']:<12.2%}", end="")
        print(f"{eval_data['accuracy_top_5']:<12.2%}", end="")
        if eval_data['perplexity'] != float('inf'):
            print(f"{eval_data['perplexity']:<12.2f}")
        else:
            print(f"{'inf':<12}")
    
    return results


def interactive_prediction_test(model: NGramPredictor):
    
    while True:
        user_input = input("\ninput the words: ").strip()
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        

        context_words = user_input.lower().split()
        
    
        predictions = model.predict_with_backoff(context_words, top_k=5)
        
        if predictions:
            print(f"\nwith context '{' '.join(context_words)}', the word next will be:")
            for i, (word, prob) in enumerate(predictions, 1):
                print(f"  {i}. {word} (proba: {prob:.4f})")
        else:
            print("cannot have predictions")


# main test
def main_test(train_data: List[str], test_data: List[str]):
  
    
    # 1. with different n
    results = compare_different_n_values(train_data, test_data, n_values=[2, 3, 4])
    
    # 2.best model
    best_n = max(results.keys(), 
                key=lambda n: results[n]['evaluation'].get('accuracy_top_1', 0))
    
    print(f"\nthe best model is: {best_n}-gramme")
    best_model = results[best_n]['model']
    
    # 3. test the interaction
    interactive_prediction_test(best_model)
    
    return results
