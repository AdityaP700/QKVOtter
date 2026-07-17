the lab2 is all about
- building an embedding layer

hence now the thing looks like this :
- How does a transformer convert token IDs into vectors?

A valid question could be :
- "How can I represent this token numerically so the neural network can process it?"

as in the first thing would be to create a list of all the token ids then with that ,
pick one index row and with that you have the
- lookup of the particular element

the question is :
- "Given token IDs produced by a tokenizer, how does the embedding layer convert them into vectors?"

hence the flow is more like :

Sentence
↓
Tokenizer
↓
Token IDs
↓
Embedding Lookup
↓
Vectors

youhuhu : used the list compression
- token_id=[vocab[word] for word in words]

The embedding layer performs no computation beyond a lookup. Each token ID acts as an index into a trainable embedding matrix, and the corresponding row is returned as that token's vector representation.

1.a) Input sentence ="I love AI"
Token IDs: [0, 1, 2]
Input Embeddings:
[[ 2.16807172 -0.52624612 -0.45550834  1.0349228 ]
 [-0.68500088 -1.1767498   0.51290774 -0.77025556]
 [-1.73403439  1.59696073  0.68661985 -0.81760448]]
Shape: (3, 4)

1.b) Input sentence = "I love dogs"
Token IDs: [0, 1, 4]

Input Embeddings:
[[-2.54937972  0.88107879 -0.56235898  0.3880601 ]
 [ 1.42471595 -1.54728246  0.41698861 -0.89686133]
 [-1.02697171 -0.84015993  0.01685436 -1.05611755]]
Shape: (3, 4)

for the experiment 2 ,we are not training the code to be accurate results

rather
An embedding vector does not become meaningful by itself. It becomes meaningful because updating it reduces the prediction loss of a downstream task.

How does an embedding stop being random?
- If an embedding participates in a prediction task, gradient descent will gradually update its row to reduce prediction loss.

Tiny Dataset
↓
Tokenizer
↓
Embedding Lookup
↓
Combine Embeddings
(your average/sum)
↓
Linear Layer
↓
Logits
↓
Cross Entropy Loss
↓
Backprop
↓
Update


Epoch 0

Embedding("AI")

[....]

Loss

2.14

Observe

Embedding row changed.

That's the entire experiment.


our main goal is to look
- how word2vec is taught


For Experiment 2

Average
or
Sum
Pick average.

Output

- Sentence Vector
- Shape
- (embedding_dim,)

here the shape of the linear layer is
- for the embedding dimension 4 and the vocab size being 20 will be 4*20

how do logits appear ?/
 -> z = XW+b

 the shape (4,) acts as the X

 the Linear weight acts as W
 - (4*20)


![alt text](image.png)

- the number with highest value of a embeded matrix are the ones the which will be choosen for the next layer

high score = most probable logit

# "Logits are the raw scores (unnormalized evidence) for every possible output token.
- "Among all the words I know, "AI" currently has the strongest evidence of being the next token."

# "How compatible is my sentence representation with every word in the vocabulary?"
- The token with the highest logit is the model's current prediction, while Softmax converts these scores into probabilities for training and sampling.

# Attention exists because averaging destroys relationships between tokens.

# Attention is a component (a mathematical layer), while the Transformer is the complete system (the deep learning architecture).

- Attention is a mathematical matrix which helps in finding relationship between words

- its components are dot products ,softmax,scaling
- used in RNNs/LSTMs

#  it computes through Q,K,V
softmax(QK`/sqrt(d_k))V :
- \(QK^{T}\) (Dot Product): Measures the geometric similarity/relevance between every word and every other word.

-  (Scaling Factor): Prevents the gradients from vanishing during training.

- Softmax: Converts the raw similarity scores into a probability distribution (weights that sum to 1).

- Multiplying by \(V\): Scales the original word meanings by those calculated attention weights.

# Transformer is like a sports car
- its a full neural network architecture
- it processes ,transforms ,and predicts data
- its components are attention ,FFNs,LayerNorm,Embedding

# a standard transformer block consists of
- positional encodings : injects order and time formation into the input embeddings

- Multi-Head Attention (MHA): Runs multiple attention operations in parallel, allowing the model to focus on different aspects of a sentence at the same time

- Residual Connections (Skip Connections): Adds the original input back to the attention output to prevent signal degradation across deep layers.

- Layer Normalization: Standardizes the mathematical scale of the outputs to ensure fast, stable training.

- Feed-Forward Networks (FFN): A set of standard dense linear layers applied to each position independently to apply non-linear transformations to the data.

# why stable_logits = logits - np.max(logits)?
- the reason is that ,logits are not always small
- what if ,softmax wants exp(1244) : it will be humongous
- it will end up being giving inf/undefined value i.e. NaN

 # Softmax only cares about relative differences.


# what is an Entropy ?
- Cross-Entropy measures the average number of bits needed to identify an event if your coding scheme is based on a wrong distribution \(Q\) rather than the true distribution \(P\).

# so what the hell is a cross entropy
- Cross-Entropy calculates how far apart your model's guess is from the absolute truth. The worse your model's guess is, the higher the Cross-Entropy score will be.

# Types of Entropies
- Binary Cross-Entropy (BCE), When to use: For yes/no, binary classification problems (e.g., Spam vs. Not Spam, Tumorous vs. Benign).

- Categorical Cross-Entropy (CCE),When to use: For multi-class classification where an item can only belong to one category (e.g., Cat vs. Dog vs. Bird).

- Sparse Categorical Cross-Entropy,When to use: Mathematically identical to CCE, but computationally optimized.Requirement: Instead of converting labels to bulky one-hot encoded arrays, you leave your labels as integers (0, 1, 2). This saves a massive amount of RAM when you have thousands of classes (like predicting the next word in a dictionary for LLMs).

- Relative Entropy (Kullback-Leibler / KL Divergence),When to use: Used in Generative AI (like VAEs or Diffusion models) to measure how much information is lost when approximating one continuous distribution with another.


# here the results for the input
"What is Japan?"

- the cross entropy calculated to be 3.3% ,that is its confident that the answer is correct i.e. "country"

hence the loss is 3.39
# True Target Word: 'country' (Token ID: 24)
<!-- Model's Assigned Probability to 'country': 0.033584
Calculated Cross-Entropy Loss: 3.393708 -->

- Cross Entropy measures how surprised the model is by the correct answer.

1) Backpropagation
Loss
↓
Gradient wrt Logits
↓
Gradient wrt W
↓
Gradient wrt Average Embedding
↓
Gradient wrt Embedding Matrix

