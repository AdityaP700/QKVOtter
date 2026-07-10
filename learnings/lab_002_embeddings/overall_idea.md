# Lab 002: Embeddings
## embedding-lookup
The lab 2 is all about building an embedding layer.
How does a transformer convert token IDs into vectors? How can I represent this token numerically so the neural network can process it?

The simple trick is embeddings. But what are embeddings?
These are just matrices - literally learned parameters. The embedding layer maps token IDs produced by a tokenizer into corresponding vectors showing its relevance.

Hence the flow is more like:
`
Sentence
↓
Tokenizer
↓
Token IDs
↓
Embedding Lookup
↓
Vectors
`
The embedding layer performs no computation beyond a lookup. Each token ID acts as an index into a trainable embedding matrix, and the corresponding row is returned as that token's vector representation.

### Experiment Results

When we ran our tokenizer against different input sentences, we generated the following outputs:

- **"I love dogs"**: [experiment_001.json](../../results/lab_002_embeddings/experiment_001.json)
- **"I love AI"**: [experiment_002.json](../../results/lab_002_embeddings/experiment_001_b.json)

The technique used for lookup is integer array indexing. For every word in words, look into the vocab dict and map out the token IDs (using list comprehension). Then, look at the embedding matrix, pull out the specific rows corresponding to the index numbers listed in `token_ids`, and stack them together to form a new matrix called `input_embeddings`.
