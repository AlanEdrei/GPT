# # First n-gram language model using a tokenized text
# 
#First Example:

# Tokens for the sentence "It shows, my dear Watson, that we are dealing
# with an exceptionally astude and dangerous man."
sample1 = ['It', 'shows', ',', 'my', 'dear', 'Watson', ',', 'that',
           'we', 'are', 'dealing', 'with', 'an', 'exceptionally',
           'astute', 'and', 'dangerous', 'man', '.']
# Tokens for the sentence "How would Lausanne do, my dear Watson?"
sample2 = ['How', 'would', 'Lausanne', 'do', ',', 'my', 'dear',
           'Watson', '?']


# First, let's make a function that splits the `tokens` sequence into its `n`-grams. We're going to use n=3



from typing import List, Tuple
def build_ngrams(tokens: List[str], n: int) -> List[Tuple[str]]:
  
    ngrams = []
    for i in range(len(tokens)-n+1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams, len(ngrams)

# Example:
build_ngrams(sample2, n=2)


# Tests:
"""
assert len(build_ngrams(sample1, n=3)) == 17
assert build_ngrams(sample1, n=3)[0] == ('It', 'shows', ',')
assert build_ngrams(sample1, n=3)[10] == ('dealing', 'with', 'an')
assert len(build_ngrams(sample1, n=2)) == 18
assert build_ngrams(sample1, n=2)[0] == ('It', 'shows')
assert build_ngrams(sample1, n=2)[10] == ('dealing', 'with')
assert len(build_ngrams(sample2, n=2)) == 8
assert build_ngrams(sample2, n=2)[0] == ('How', 'would')
assert build_ngrams(sample2, n=2)[7] == ('Watson', '?')
"""

# With the current function, there's no way to know whether an n-gram is at the beginning, middle, or end of the sequence. To overcome this problem, n-gram language models often include special "beginning-of-string" (BOS) and "end-of-string" (EOS) control tokens.
# 
# Write a new version of your `build_ngrams` function that includes these control tokens. For instance, when `tokens=sample1` and `n=3`, our new function should return:
# 
# ```python
# [('<BOS>', '<BOS>', 'It'),
#  ('<BOS>', 'It', 'shows'),
#  ('It', 'shows', ','),
#  ('shows', ',', 'my'),
#  (',', 'my', 'dear'),
#  ...,
#  ('dangerous', 'man', '.'),
#  ('man', '.', '<EOS>'),
#  ('.', '<EOS>', '<EOS>')]
# ```



BOS = '<BOS>'
EOS = '<EOS>'

def build_ngrams_ctrl(tokens: List[str], n: int) -> List[Tuple[str]]:
    # your code here
    tokens = [BOS] * (n-1) + tokens + [EOS] * (n-1)
    ngrams=[]
    
    for i in range (len(tokens)-n+1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

# Example:
build_ngrams_ctrl(sample1, n=3)



# Tests:
"""
assert len(build_ngrams_ctrl(sample1, n=3)) == 21
assert build_ngrams_ctrl(sample1, n=3)[0] == ('<BOS>', '<BOS>', 'It')
assert build_ngrams_ctrl(sample1, n=3)[10] == ('we', 'are', 'dealing')
assert len(build_ngrams_ctrl(sample1, n=2)) == 20
assert build_ngrams_ctrl(sample1, n=2)[0] == ('<BOS>', 'It')
assert build_ngrams_ctrl(sample1, n=2)[10] == ('are', 'dealing')
assert len(build_ngrams_ctrl(sample2, n=2)) == 10
assert build_ngrams_ctrl(sample2, n=2)[0] == ('<BOS>', 'How')
assert build_ngrams_ctrl(sample2, n=2)[9] == ('?', '<EOS>')
"""

# Now that n-grams are builded, we have almost everything we need to build an n-gram language model.

# To compute Maximum Likelihood Estimations, we first need to count the number of times each word follows an n-gram of size `n-1`. 


from typing import Dict
def count_ngrams(texts: List[List[str]], n: int) -> Dict[Tuple[str, ...], Dict[str, int]]:
    # your code here
    # Be sure to use your build_ngrams_ctrl implementation
    ngram_counts = {}

    for text_num, text in enumerate(texts, start=1):
        print(f"Procesando texto {text_num}...")
        ngrams = build_ngrams_ctrl(text, n)
        print(f"Generados {len(ngrams)} n-gramas de tamaño {n}")
        
        for ngram_num, ngram in enumerate(ngrams, start=1):
            prefix = tuple(ngram[:-1])  # n-1 gram
            word = ngram[-1]  # Última palabra en el n-grama
            

            if prefix not in ngram_counts:
                ngram_counts[prefix] = {}
            if word not in ngram_counts[prefix]:
                ngram_counts[prefix][word] = 1
            else:
                ngram_counts[prefix][word] += 1
                
    return ngram_counts

# Example:
count_ngrams([sample1, sample2], n=3)


# Como se puede notar, el numero de engramas no coincide con el numero de entradas del diccionario. Aquí pueden pasar dos cosas: 
# 
# 1. Que ambas entradas compartan un mismo n-grama, con la misma palabra final como en:('my', 'dear'): {'Watson': 2}
# 2. Que para un engrama, el mismo n-grama venga seguido de palabras distintas:('dear', 'Watson'): {',': 1, '?': 1}




# Tests:
"""
assert len(count_ngrams([sample1, sample2], n=3)) == 28
assert len(count_ngrams([sample1, sample2], n=3)['<BOS>', '<BOS>']) == 2
assert count_ngrams([sample1, sample2], n=3)['<BOS>', '<BOS>']['It'] == 1
assert count_ngrams([sample1, sample2], n=3)['<BOS>', '<BOS>']['How'] == 1
assert count_ngrams([sample1, sample2], n=3)['my', 'dear']['Watson'] == 2
assert len(count_ngrams([sample1, sample2], n=2)) == 24
assert len(count_ngrams([sample1, sample2], n=2)['<BOS>',]) == 2
assert count_ngrams([sample1, sample2], n=2)['<BOS>',]['It'] == 1
assert count_ngrams([sample1, sample2], n=2)['<BOS>',]['How'] == 1
assert count_ngrams([sample1, sample2], n=2)['dear',]['Watson'] == 2
"""

#  The last step is to convert the counts into probability estimates.
# 
# When `texts=[sample1, sample2]` and `n=3`, your function should return:
# 
# ```python
# {
#     ('<BOS>', '<BOS>'): {'It': 0.5, 'How': 0.5},
#     ('<BOS>', 'It'): {'shows': 1.0},
#     ('<BOS>', 'How'): {'would': 1.0},
#     ...
#     ('my', 'dear'): {'Watson': 1.0},
#     ('dear', 'Watson'): {',': 0.5, '?': 0.5},
#     ...
# }
# ```


from typing import Dict
def build_ngram_model(texts: List[List[str]], n: int) -> Dict[Tuple[str, ...], Dict[str, float]]:
    
   
 
    ngram_counts = count_ngrams(texts, n)
    ngram_model = {}

    for prefix, word_counts in ngram_counts.items():
        total_count = sum(word_counts.values())
        probabilities = {word: count / total_count for word, count in word_counts.items()}
        ngram_model[prefix] = probabilities

    return ngram_model


# Example:
build_ngram_model([sample1, sample2], n=3)





# Tests:
"""
assert build_ngram_model([sample1, sample2], n=3)['<BOS>', '<BOS>']['It'] == 0.5
assert build_ngram_model([sample1, sample2], n=3)['<BOS>', '<BOS>']['How'] == 0.5
assert build_ngram_model([sample1, sample2], n=3)['my', 'dear']['Watson'] == 1.0
assert build_ngram_model([sample1, sample2], n=2)['<BOS>',]['It'] == 0.5
assert build_ngram_model([sample1, sample2], n=2)['<BOS>',]['How'] == 0.5
assert build_ngram_model([sample1, sample2], n=2)['dear',]['Watson'] == 1.0
"""

# A language model built from only a few sentences is not very informative. Let's scale up and see what your language model looks like when we train on the complete works of Sir Arthur Conon Doyle!



full_text = []
with open('arthur-conan-doyle.tok.train.txt', 'rt') as fin:
    for line in fin:
        full_text.append(list(line.split()))
model = build_ngram_model(full_text, n=3)



for prefix in [(BOS, BOS), (BOS, 'It'), ('It', 'was'), ('my', 'dear')]:
    print(*prefix)
    sorted_probs = sorted(model[prefix].items(), key=lambda x: -x[1])
    for k, v in sorted_probs[:5]:
        print(f'\t{k}\t{v:.4f}')
    print(f'\t[{len(sorted_probs)-5} more...]')



