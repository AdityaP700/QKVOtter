
- i have observed that for different input tokens how the tokenizer behaves.

- surprisingly the results are different

as of now i am curious enough to know
how one token attends to other tokens

the simple trick is the embeddings
but what is embeddings??


anything magical? no right
at the end all these things are mathematical things

hence what is embeddings

these are just matrices

about how one token id is being
transformed into something showing its relevance as in the form of a vector

and these vectors are used to identify

which token id might be attending to any other tokens

hence whats the pipeline to be created ?

- first of all ,build an embedding matrix

but from  Where do these values come from?

- embeddings are literally learned parameters.

More token splitting

↓

Longer sequences

↓

More attention cost

Large vocabulary

250000

↓

Shorter sequences

↓

Huge embedding matrix

↓

Huge output layer

↓

More memory


well backpropagation
doesnt says
- change embeddings

it says to find every parameter that contributed to this loss.

∂Weight
∂Loss
	​


Read it as

"If this weight changes a tiny bit,
how much does the loss change?"


"Where are these gradients calculated from?"

From the loss.

Prediction

↓

Loss

↓

Backpropagation

↓

Gradients

↓

Parameter updates


think of backpropagation

Token IDs

↓

Embedding

↓

Attention

↓

Feed Forward

↓

Logits

↓

Softmax

↓

Loss

During the forward pass, information flows downward.

When training, we ask

"Who caused this loss?"

So the error flows backward:

Loss

↑

Softmax

↑

Linear layer

↑

Attention

↑

Embedding

That's why it's called back-propagation.

Each layer receives a gradient from the layer after it, computes how much it contributed to the error, and passes gradients further backward using the chain rule from calculus.

for our experiment we will be adding the embeddings inorder to get a simplified results

we are not comparing vectors
we are comparing predictions
These are logits
not to compare vectors

Input Tokens
↓
Embedding Layer
↓
Transformer
↓
Vocabulary Scores
↓
Loss

The embedding layer is just another trainable layer.

vocab_size =5
st_vec=4

Backpropagation doesn't care whether you're training XOR, GPT, or a vision model. It always updates parameters in the same way:


![alt text](image.png)

Suppose your model outputs
[0.2, 1.3, 4.8, 0.9, 2.1]
and the correct next word is
AI
(which has token ID 2).

Embedding Lookup
        ↓
Linear Layer
        ↓
Vocabulary Logits
        ↓
Cross Entropy Loss
        ↓
Gradient
        ↓
Update:
    • Embedding Matrix
    • Linear Layer

![alt text](image-1.png)

The only difference is

BCE distributes probability over 2 classes
Cross Entropy distributes probability over V classes (entire vocabulary)

also :
- The number of output neurons equals the number of classes you want to predict.

Suppose the prompt is

"The capital of France is"

GPT literally computes

Paris        -> 12.7
London       -> 3.1
Berlin       -> 2.8
Tokyo        -> 0.4
...
<all 50k tokens>

It doesn't directly output

Paris

It first scores every token.

Then

Logits
↓
Softmax
↓
Probability Distribution
↓
Sampling / Greedy
↓
Next Token

![alt text](image-2.png)

For GPT-2:
50257 × 768

Now the output layer is
768 × 50257

Do you notice something?

Embedding Matrix
50257 × 768

vs

Output Matrix
768 × 50257

Many LLMs actually share these weights.

This is called weight tying.

Instead of learning two separate huge matrices, they reuse the same knowledge.

Conceptually

Token ID
↓
Embedding Matrix
↓
768-dimensional vector

...

Transformer

...

768-dimensional hidden state
↓
Embedding Matrixᵀ
↓
Vocabulary logits

✅ Tokenizer → Token IDs
✅ Token IDs → Embedding lookup
✅ Why the embedding matrix is (V × d)
✅ Why the output layer is (d × V)
✅ Why GPT predicts one score per vocabulary token
✅ Why Cross Entropy is used
✅ How gradients eventually flow back into the embedding matrix