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