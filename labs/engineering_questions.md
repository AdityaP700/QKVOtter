
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

