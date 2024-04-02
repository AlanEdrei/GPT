# # Text Generation with an N-Gram Language Model using  Greedy Search and Sampling.

# base implementation:

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

    def get_prob_dist(self, context):
        # Take only the n-1 most recent context (Markov Assumption)
        context = tuple(context[-self.n+1:])
        # Add <BOS> tokens if the context is too short, i.e., it's at the start of the sequence
        while len(context) < (self.n-1):
            context = (BOS,) + context
        # Handle words that were not encountered during the training by replacing them with a special <OOV> token
        context = tuple((c if c in self.V else OOV) for c in context)
        if context in self.model:
            # Compute the probability distribution using a Maximum Likelihood Estimation and Laplace Smoothing
            norm = sum(self.model[context].values()) + self.smoothing * len(self.V)
            prob_dist = {k: (c + self.smoothing) / norm for k, c in self.model[context].items()}
            for word in self.V - prob_dist.keys():
                prob_dist[word] = self.smoothing / norm
        else:
            # Simplified formula if we never encountered this context; the probability of all tokens is uniform
            prob = 1 / len(self.V)
            prob_dist = {k: prob for k in self.V}
        prob_dist = dict(sorted(prob_dist.items(), key=lambda x: (-x[1], x[0])))
        return prob_dist




# Load pre-built n-gram languae models
model_unigram = NGramLM('arthur-conan-doyle.tok.train.n1.pkl')
model_bigram = NGramLM('arthur-conan-doyle.tok.train.n2.pkl')
model_trigram = NGramLM('arthur-conan-doyle.tok.train.n3.pkl')
model_4gram = NGramLM('arthur-conan-doyle.tok.train.n4.pkl')
model_5gram = NGramLM('arthur-conan-doyle.tok.train.n5.pkl')


# Let's take a look at some of the probability distributions, just to check what we have.

model_bigram.get_prob_dist(['elemental'])

model_bigram.get_prob_dist(['.'])

model_trigram.get_prob_dist(["no",",","I"])


# We'll start with a simple greedy generation approach. 

from typing import List
def greedy_generation(model: NGramLM, context: List[str], max_length: int = 100) -> List[str]:
    
    generated_sequence = context[:]
    while len(generated_sequence) < max_length:
            prob_dist = model.get_prob_dist(generated_sequence)
            next_word = max(prob_dist, key=prob_dist.get)
            if next_word == EOS:
                break
            generated_sequence.append(next_word)
    return generated_sequence

greedy_generation(model_4gram, ["elemental",",","my","dear"])



# 
# As we can see, this generation is deterministic and not very interesting
# Even if we consider trying different model types (unigram, bigram, trigram, 4-gram, and 5-gram)


from typing import List
def greedy_generation(model: NGramLM, context: List[str], max_length: int = 100) -> List[str]:
    
    generated_sequence = context[:]
    while len(generated_sequence) < max_length:
            prob_dist = model.get_prob_dist(generated_sequence)
            next_word = max(prob_dist, key=prob_dist.get)
            if next_word == EOS:
                break
            generated_sequence.append(next_word)
    return generated_sequence

greedy_generation(model_trigram, ["elemental",",","my","dear"])


# Now it's time to implement sampling.


from typing import List
import random
def sampling_generation(model: NGramLM, context: List[str], max_length: int = 100, topk=10) -> List[str]:
    generated_sequence = context[:]
    while len(generated_sequence) < max_length:
        prob_dist = model.get_prob_dist(generated_sequence)
        # Reduce the candidate set to only the topk highest-probability items
        topk_probs = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)[:topk]
        topk_words = [word for word, prob in topk_probs]
        # Use random.choices to sample the next word from the topk words
        next_word = random.choices(topk_words, weights=[prob for word, prob in topk_probs], k=1)[0]
        if next_word == EOS:
            break
        generated_sequence.append(next_word)
    return generated_sequence

sampling_generation(model_trigram, ['""', 'My', 'dear', 'Watson'])
