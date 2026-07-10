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

