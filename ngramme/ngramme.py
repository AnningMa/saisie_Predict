from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import math
import sys
sys.path.append("C:\\Users\\Agnes\\Desktop\\talproject")  
import data  


class NGramStorage:
    """存储和管理所有n-gram数据的类"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls) #如果没有（即第一次创建），就调用父类的 __new__ 方法来创建一个实例，并存储在 cls._instance 中。
        return cls._instance  #如果已经创建过了，就直接返回已有的实例。
    
    def __init__(self):
        if not NGramStorage._initialized:  #储存：context和next_word, processed_sentences
            self.ngram_counts = {}  # {n: {ngram: {next_word: count}}} 
            self.context_counts = {}  # {n: {context: total_count}} 每个context及其出现总数
            self.vocabulary = set()  ## 所有出现过的词（包括<START>和<END>）
            self.max_built_n = 0  # 记录已经构建到第几阶
            self.processed_sentences = []  # 存储处理过的句子
            NGramStorage._initialized = True
    
    def build_ngrams_incremental(self, sentences: List[str], target_n: int):
        #从 1-gram 到 max_n-gram，构建所有可能的 n-gram。
        # 如果是第一次构建或者需要更新句子
        if not self.processed_sentences:
            self.processed_sentences = self._process_sentences(sentences, target_n)
        
        # 增量构建从 max_built_n+1 到 target_n 的n-gram
        for n in range(self.max_built_n + 1, target_n + 1):
            print(f"Building {n}-grams...")
            if n not in self.ngram_counts:
                self.ngram_counts[n] = defaultdict(Counter)
                self.context_counts[n] = defaultdict(int)
            
            for padded_words in self.processed_sentences:
                for i in range(len(padded_words) - n + 1):  #如果句子有 L 个词，那么就有 L - n + 1 个 n-gram。
                    if n == 1:
                        word = padded_words[i]
                        self.ngram_counts[1][()][word] += 1  #unigram没有上下文
                        self.context_counts[1][()] += 1  
                    else:
                        context = tuple(padded_words[i:i + n - 1])
                        next_word = padded_words[i + n - 1]
                        self.ngram_counts[n][context][next_word] += 1
                        self.context_counts[n][context] += 1
        
        self.max_built_n = max(self.max_built_n, target_n) #避免重复建已经建过的 n-gram
        print(f"Built n-grams up to {self.max_built_n}. ")
        
    def build_single_ngram(self, sentences: List[str], target_n: int):
        """独立构建指定阶数的n-gram，不依赖低阶n-gram"""
        print(f"Building {target_n}-gram independently...")
        
        # 处理句子（如果还没处理过）
        if not self.processed_sentences:
            self.processed_sentences = self._process_sentences(sentences, target_n)
            
            for sentence_words in self.processed_sentences:
                self.vocabulary.update(sentence_words)
        
        # 初始化指定阶数的存储
        if target_n not in self.ngram_counts:
            self.ngram_counts[target_n] = defaultdict(Counter)
            self.context_counts[target_n] = defaultdict(int)
        
        # 只构建target_n阶的n-gram
        for padded_words in self.processed_sentences:
            for i in range(len(padded_words) - target_n + 1):
                if target_n == 1:
                    word = padded_words[i]
                    self.ngram_counts[1][()][word] += 1
                    self.context_counts[1][()] += 1
                else:
                    context = tuple(padded_words[i:i + target_n - 1])
                    next_word = padded_words[i + target_n - 1]
                    self.ngram_counts[target_n][context][next_word] += 1
                    self.context_counts[target_n][context] += 1
        
        # 更新max_built_n（如果这是最高阶）
        if target_n > self.max_built_n:
            self.max_built_n = target_n
            
        print(f"Built {target_n}-gram. Vocabulary size: {len(self.vocabulary)}")
    
    def _process_sentences(self, sentences: List[str], max_n: int) -> List[List[str]]:
        processed = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 0:
                padded_words = ['<START>'] * (max_n - 1) + words + ['<END>']
                processed.append(padded_words)
        return processed
    
    def get_ngram_data(self, n: int):
        """获取指定n的gram数据"""
        if n not in self.ngram_counts:
            raise ValueError(f"N-gram of order {n} not built yet. Current max: {self.max_built_n}")
        return self.ngram_counts.get(n, {}), self.context_counts.get(n, {})


class NGram:
    
    def __init__(self, n: int, storage: NGramStorage, smoothing_method: str = 'laplace', min_count: int = 1):
        self.n = n
        self.smoothing_method = smoothing_method
        self.min_count = min_count
        self.storage = storage
        
        # 确保所需阶数的n-gram已经构建
        if n not in storage.ngram_counts:
            raise ValueError(f"N-gram of order {n} not available. Use build_ngrams_incremental() first.")
        
        # 获取数据
        self.ngram_counts, self.context_counts = storage.get_ngram_data(n)
        self.vocabulary = storage.vocabulary
        
        # 应用最小计数过滤
        if min_count > 1:
            self._apply_min_count_filtering()
    
    def _apply_min_count_filtering(self):
        """应用最小计数过滤"""
        if self.n == 1:
            return
            
        filtered_counts = defaultdict(Counter)
        filtered_context_counts = defaultdict(int)
        
        for context, word_counts in self.ngram_counts.items():
            total_count = sum(word_counts.values())
            if total_count >= self.min_count:
                filtered_word_counts = Counter()
                for word, count in word_counts.items():
                    if count >= self.min_count:
                        filtered_word_counts[word] = count
                
                if filtered_word_counts:
                    filtered_counts[context] = filtered_word_counts
                    filtered_context_counts[context] = sum(filtered_word_counts.values())
        
        self.ngram_counts = filtered_counts
        self.context_counts = filtered_context_counts
    
    def get_probability(self, context: Tuple[str, ...], word: str) -> float:
        """获取平滑概率"""
        if self.n == 1:
            word_count = self.ngram_counts[()][word]
            total_count = self.context_counts[()]
        else:
            if context not in self.ngram_counts:
                return 0.0
            
            word_count = self.ngram_counts[context][word]
            total_count = self.context_counts[context]
        
        if total_count == 0:
            return 0.0
        
        if self.smoothing_method == 'laplace':
            return (word_count + 1) / (total_count + len(self.vocabulary))
        else:
            return word_count / total_count if total_count > 0 else 0.0
    
    def get_candidates(self, context: Tuple[str, ...], top_k: int = 10) -> List[Tuple[str, float]]:
        """获取候选词及其概率"""
        if self.n == 1:
            word_counts = self.ngram_counts[()]
        else:
            if context not in self.ngram_counts:
                return []
            word_counts = self.ngram_counts[context]
        
        predictions = []
        for word, count in word_counts.most_common(top_k * 2):
            if word != '<END>':
                prob = self.get_probability(context, word)
                predictions.append((word, prob))
                if len(predictions) >= top_k:
                    break
        
        return predictions


class NGramPredictor:
    """N-gram预测器类"""
    
    def __init__(self, n: int, storage: NGramStorage, smoothing_method: str = 'laplace', 
                 min_count: int = 1, use_interpolation: bool = False):
        self.n = n
        self.use_interpolation = use_interpolation
        self.storage = storage
        
        # 确保需要的所有阶数都已构建
        if n > storage.max_built_n:
            raise ValueError(f"N-gram of order {n} not available. Current max: {storage.max_built_n}")
        
        # 创建所有阶数的n-gram模型
        self.ngrams = {}
        for i in range(1, n + 1):
            self.ngrams[i] = NGram(i, storage, smoothing_method, min_count)
        
        # 插值权重（简化版本）
        if use_interpolation:
            # 简单的线性插值权重
            total = sum(range(1, n + 1))
            self.interpolation_weights = {i: i / total for i in range(1, n + 1)}
        else:
            self.interpolation_weights = {n: 1.0}
    
    def predict_next_words(self, context_words: List[str], top_k: int = 6) -> List[Tuple[str, float]]:
        """预测下一个词"""
        if self.use_interpolation:           
            return self._predict_with_interpolation(context_words, top_k)
        else:
            return self._predict_with_backoff(context_words, top_k)
    
    
    def _predict_with_backoff(self, context_words: List[str], top_k: int) -> List[Tuple[str, float]]:
        """使用回退策略预测"""
        for n in range(min(self.n, len(context_words) + 1), 0, -1):
            if n == 1:
                context = ()
            else:
                if len(context_words) >= n - 1:
                    context = tuple(context_words[-(n-1):])
                else:
                    context = tuple(['<START>'] * (n - 1 - len(context_words)) + context_words)
            
            predictions = self.ngrams[n].get_candidates(context, top_k)
            if predictions:
                return predictions
        
        return []
    
    def _predict_with_interpolation(self, context_words: List[str], top_k: int) -> List[Tuple[str, float]]:
        """使用插值预测"""
        # 收集所有候选词
        candidate_words = set()
        for n in range(1, self.n + 1):
            if n == 1:
                context = ()
            else:
                if len(context_words) >= n - 1:
                    context = tuple(context_words[-(n-1):])
                else:
                    context = tuple(['<START>'] * (n - 1 - len(context_words)) + context_words)
            
            candidates = self.ngrams[n].get_candidates(context, top_k * 2)
            candidate_words.update([word for word, _ in candidates])
        
        # 计算插值概率
        word_probs = {}
        for word in candidate_words:
            interpolated_prob = 0.0
            for n in range(1, self.n + 1):
                if n == 1:
                    context = ()
                else:
                    if len(context_words) >= n - 1:
                        context = tuple(context_words[-(n-1):])
                    else:
                        context = tuple(['<START>'] * (n - 1 - len(context_words)) + context_words)
                
                prob = self.ngrams[n].get_probability(context, word)
                weight = self.interpolation_weights.get(n, 0)
                interpolated_prob += weight * prob
            
            word_probs[word] = interpolated_prob
        
        return sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def evaluate(self, test_sentences: List[str]) -> Dict[str, float]:
        """简化的评估函数"""
        total_predictions = 0
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        
        for sentence in test_sentences:
            words = sentence.split() if isinstance(sentence, str) else sentence
            if len(words) < self.n:
                continue
            
            padded_words = ['<START>'] * (self.n - 1) + words
            
            for i in range(self.n - 1, len(padded_words)):
                context = padded_words[i - (self.n - 1):i]
                actual_word = padded_words[i]
                
                predictions = self.predict_next_words(context, 5)
                
                if predictions:
                    total_predictions += 1
                    predicted_words = [word for word, _ in predictions]
                    
                    if len(predicted_words) > 0 and actual_word == predicted_words[0]:
                        correct_top1 += 1
                    if len(predicted_words) >= 3 and actual_word in predicted_words[:3]:
                        correct_top3 += 1
                    if len(predicted_words) >= 5 and actual_word in predicted_words[:5]:
                        correct_top5 += 1
        
        if total_predictions == 0:
            return {'top1': 0.0, 'top3': 0.0, 'top5': 0.0, 'total': 0}
        
        return {
            'top1': correct_top1 / total_predictions,
            'top3': correct_top3 / total_predictions,
            'top5': correct_top5 / total_predictions,
            'total': total_predictions
        }


def create_model(train_data: List[str], n: int) -> NGramPredictor:
    """创建单个n-gram模型（增量式）"""
    storage = NGramStorage()
    
    # 增量构建到目标n阶
    storage.build_ngrams_incremental(train_data, n)
    
    # 创建预测器
    model = NGramPredictor(
        n=n, 
        storage=storage, 
        smoothing_method='laplace',
        min_count=5,
        use_interpolation=True if n > 1 else False
    )
    
    return model


def create_models(train_data: List[str], max_n: int = 4) -> Dict[int, NGramPredictor]:
    """创建多个n-gram模型（增量式，复用低阶n-gram）"""
    storage = NGramStorage()
    models = {}
    
    # 逐步构建，每次只计算新的阶数
    for n in range(1, max_n + 1):
        print(f"\nCreating {n}-gram model...")
        storage.build_ngrams_incremental(train_data, n)
        
        models[n] = NGramPredictor(
            n=n, 
            storage=storage, 
            smoothing_method='laplace',
            min_count=5,
            use_interpolation=True if n > 1 else False
        )
    
    return models


def evaluate_models(models: Dict[int, NGramPredictor], test_data: List[str]):
    """评估所有模型"""
    print(f"{'Model':<10}{'Top-1':<10}{'Top-3':<10}{'Top-5':<10}{'Total':<10}")
    print("-" * 50)
    
    results = {}
    for n, model in models.items():
        result = model.evaluate(test_data)
        results[n] = result
        print(f"{n}-gram    {result['top1']:<10.2%}{result['top3']:<10.2%}{result['top5']:<10.2%}{result['total']:<10}")
    
    return results


def interactive_test(model: NGramPredictor):
    """交互式测试"""
    print(f"\nInteractive Testing - {model.n}-gram Model")
    print("Enter context (type 'quit' to exit):")
    
    while True:
        user_input = input("\nContext: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input:
            continue
        
        context_words = user_input.split()
        predictions = model.predict_next_words(context_words, 5)
        
        print(f"Predictions:")
        if predictions:
            for i, (word, prob) in enumerate(predictions, 1):
                print(f"  {i}. {word} (prob: {prob:.4f})")
        else:
            print("  No predictions available")


def main():
    """主函数"""
    # 加载数据
    train_sentences = data.train_data
    test_sentences = data.test_data
    
    # 创建模型（只需要构建一次n-gram数据）
    models = create_models(train_sentences, max_n=10)
    
    # 评估所有模型
    results = evaluate_models(models, test_sentences)
    
    # 选择最佳模型进行交互测试
    best_model = models[2]  # 默认选择2-gram
    interactive_test(best_model)
    
    return models, results


if __name__ == "__main__":
    models, results = main()