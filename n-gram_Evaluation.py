# Evaluating an N-Gram Language Model using perplexity.
# 
# implementation to test:


import pickle
BOS = '<BOS>'
EOS = '<EOS>'
OOV = '<OOV>'
class NGramLM:
    def __init__(self, path, smoothing=0.001, verbose=False):
        with open(path, 'rb') as fin:
            data = pickle.load(fin)
        self.n = data['n']
        self.V = set(data['V'])
        self.model = data['model']
        self.smoothing = smoothing
        self.verbose = verbose

    def get_prob(self, context, token):
        # Take only the n-1 most recent context (Markov Assumption)
        context = tuple(context[-self.n+1:])
        # Add <BOS> tokens if the context is too short, i.e., it's at the start of the sequence
        while len(context) < (self.n-1):
            context = (BOS,) + context
        # Handle words that were not encountered during the training by replacing them with a special <OOV> token
        context = tuple((c if c in self.V else OOV) for c in context)
        if token not in self.V:
            token = OOV
        if context in self.model:
            # Compute the probability using a Maximum Likelihood Estimation and Laplace Smoothing
            count = self.model[context].get(token, 0)
            prob = (count + self.smoothing) / (sum(self.model[context].values()) + self.smoothing * len(self.V))
        else:
            # Simplified formula if we never encountered this context; the probability of all tokens is uniform
            prob = 1 / len(self.V)
        # Optional logging
        if self.verbose:
            print(f'{prob:.4n}', *context, '->', token)
        return prob



# We load pre-built n-gram languae models
model_unigram = NGramLM('arthur-conan-doyle.tok.train.n1.pkl')
model_bigram = NGramLM('arthur-conan-doyle.tok.train.n2.pkl')
model_trigram = NGramLM('arthur-conan-doyle.tok.train.n3.pkl')
model_4gram = NGramLM('arthur-conan-doyle.tok.train.n4.pkl')
model_5gram = NGramLM('arthur-conan-doyle.tok.train.n5.pkl')


# Now we're going to see how well these models fit our data. We'll use Perplexity for this calculation. (perplexity = 2^{\frac{-1}{n}\sum \log_2(P(w_i|w_{<i}))})


import math


from typing import List, Tuple

def perplexity(model: NGramLM, texts: List[Tuple[str]]) -> float:
    total_log_prob = 0
    total_words = 0
    
    for text in texts:
        for i in range(len(text)):
            context = text[:i]
            token = text[i]
            total_log_prob += math.log2(model.get_prob(context, token))
            total_words += 1
    
    avg_log_prob = total_log_prob / total_words
    perplexity_score = 2 ** (-avg_log_prob)
    
    return perplexity_score
# Example:
perplexity_score = perplexity(model_unigram, [('My', 'dear', 'Watson')])
print("Perplexity:", perplexity_score)



# Tests
assert round(perplexity(model_unigram, [('My', 'dear', 'Watson')])) == 7531
assert round(perplexity(model_bigram, [('My', 'dear', 'Watson')])) == 24
assert round(perplexity(model_trigram, [('My', 'dear', 'Watson')])) == 521


# Now let's see how well the model fits a held-out test set.

toks_test = []
with open('arthur-conan-doyle.tok.test.txt', 'rt') as fin:
    for line in fin:
        toks_test.append(list(line.split()))
        
# Calcular la perplejidad para cada modelo.

models = [model_unigram, model_bigram, model_trigram, model_4gram, model_5gram]
n_values = [1, 2, 3, 4, 5]

for model, n in zip(models, n_values):
    perplexity_score = perplexity(model, toks_test)
    print(f"Perplexity for {n}-gram model: {perplexity_score}")        
        






