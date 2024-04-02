# # Language Generation with Transformers
# When predicting the next token, a GPT model can give us a score for all possible next tokens. We can use those probabilities to generate new text, potentially by selecting the most likely next token or by sampling using the probabilities. Let's see how that works.
# Let's say that we want to generate more text after the sequence below:

text = 'The quick brown fox jumped over'

# We'll need to load the tokenizer and model for `distilgpt2`.

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

# As before, we use the tokenizer to tokenize the text and convert each token to its token ID. We will use the `.encode` function to get the token IDs back as a Python list as they are easier to manipulate. We'll want to add extra token IDs that we've generated!

input_ids = tokenizer.encode(text)
input_ids

#We can use the `tokenizer.decode` function to turn the token IDs back into text. This will be useful after we've generated further token IDs to add on the end

tokenizer.decode(input_ids)

#Now let's run the token IDs through the `distilgpt2` model and get the probabilities of the next token

import torch # We'll load Pytorch so we can convert a list to a tensor
from scipy.special import softmax

as_tensor = torch.tensor(input_ids).reshape(1,-1) # This converts the token ID list to a tensor
output = model(input_ids=as_tensor) # We pass it into the model
next_token_scores = output.logits[0,-1,:].detach().numpy() # We get the scores for next token and the end of the sequence (token index=-1)
next_token_probs = softmax(next_token_scores) # And we apply a softmax function

next_token_probs.shape

# Now we've got the probabilities for all possible 50257 tokens to be after our input text sequence.

# Let's get the one with the highest probability. For that we can use the `argmax` function.

next_token_id = next_token_probs.argmax()
next_token_id

#the token with ID=262 has the highest probability. But what token is that? `tokenizer.decode` can tell us:

tokenizer.decode(next_token_id)

#Let's calculate the next eight tokens after `input_ids` (including the one we calculated above). 

import torch
from scipy.special import softmax

# Starting with the given input_ids
input_ids = [464, 2068, 7586, 21831, 11687, 625]

# Loop to calculate the next eight tokens
for _ in range(8):
    # Convert token ID list to a tensor
    as_tensor = torch.tensor(input_ids).reshape(1, -1)
    
    # Pass it into the model
    output = model(input_ids=as_tensor)
    
    # Get the scores for the next token at the end of the sequence
    next_token_scores = output.logits[0, -1, :].detach().numpy()
    
    # Apply softmax function to get probabilities
    next_token_probs = softmax(next_token_scores)
    
    # Get the token with the highest probability
    next_token_id = next_token_probs.argmax()
    
    # Append the new token ID to input_ids list
    input_ids.append(next_token_id)
    
    # Decode the new token ID and print it
    print(tokenizer.decode(next_token_id))

# Final sequence of token IDs
print("Final sequence of token IDs:", input_ids)

#Now picking the token with highest probability every time can often create quite boring text. Sampling from the tokens can generate more interesting text. Sampling uses the probabilities as weights so that words with higher probabilities are more likely to be chosen. Let's see how that works:

import numpy as np # We're using numpy to use its argmax function

next_token_probs = np.array([0.1, 0.2, 0.5, 0.3])

#As we saw above, we can use `argmax` that tells us the index of the highest value. In this case, it's index=2

next_token_probs.argmax()

#However, let's say we want to sample randomly from the possible token indices (`[0, 1, 2, 3]`). First, let's create that list to sample from:

indices = list(range(len(next_token_probs)))
indices

#We could use the choices function to pick a single token ID with all four being equally likely to be chosen:

import random

for i in range (8):
    next_token_id = random.choices(indices, k=1)[0]
    print(next_token_id)

# Or we could provide weights, such that some of the tokens are more likely to be chosen than others. In this case, we provide `next_token_probs` as weights.

for i in range (8):
    next_token_id = random.choices(indices, k=1, weights=next_token_probs)[0]
    print(next_token_id)


# The HuggingFace library provides a `text-generation` pipeline to generate text.
# For example, here is how to run it and request 30 extra tokens and 5 different generations.

from transformers import pipeline
generator = pipeline('text-generation', model="distilgpt2")
generator("Hello, I'm a language model,", max_new_tokens=30, num_return_sequences=5)

#There are a lot of different options, including controlling how sampling is done. If we wanted to not do sampling, we could turn it off with `do_sample=False`.
generator("Hello, I'm a language model,", max_new_tokens=30, do_sample=False)

#Or turn it on but tell it to only sample from the 10 most likely tokens, we can use `do_sample=True` and `top_k=10`
generator("Hello, I'm a language model,", max_new_tokens=30, do_sample=True, top_k=10)

