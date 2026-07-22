the question is ,why is averaging  embeddings fundamentally broken??

Example
- Dog bites man
embeddings -> average -> vector A

- Man bites dog
embeddings-> average -> vector B

Embeddings lose relationships

hence attention


now the question is ,even if i have assigned the index ,the thing is :
- How could a neural network learn these weights instead of me writing them?

there should be a mechanism that lets each word decide how much every other words matters?


Average pooling says
- Equal weights.

my experiment says
- Manual weights.

Self-attention says

- Learn the weights for every sentence automatically.

- Query : What the word is looking for (its question).

- Key (\(K\)): What the word offers (its label or topic tag).

- Value (\(V\)): The actual meaning/content of the word if it matches.

- "I" broadcasts: "I am a pronoun/subject.
- "love" broadcasts: "I am an action/verb requiring an object."
- "dogs" broadcasts: "I am a plural noun/object."

Instead of me choosing these scores, how can the model compute them by itself?