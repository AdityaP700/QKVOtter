# QKVOTter

(Yes, I named it QKVOTter purely because it sounded cute in my head and I am dealing with the architecture behind the transformer model)

I'm building this from scratch because I actually want to understand how a Transformer works, rather than just pretending I do by throwing JSON at an OpenAI endpoint.

I throw around buzzwords like "Self-Attention," "KV Cache," and "Context Windows" at parties to sound smart, but under the hood, I'm really just wrestling with the same core mathematical problems that dictate how machines parse human language:

How do I translate messy, infinite human language into a strictly optimized sequence of numbers that a GPU won't choke on?

How does a model figure out what a word means based purely on the geometry of a high-dimensional vector space?

How does a system learn grammar, reasoning, and logic strictly as a byproduct of trying to guess the next word in a sentence?

I don't want to just `pip install transformers` and let HuggingFace sweep these problems under the rug. I want to solve them manually, painfully, and step-by-step.

## The Journey So Far (The Messy Labs)

Here is what I've figured out so far while banging my head against the keyboard:

**Lab 001: The Tokenizer Problem**
How do tokenizer design choices influence what ultimately reaches the transformer? (Spoiler: A bad tokenizer explodes the token count, which geometrically destroys the attention compute budget. I compared a custom, naïve BPE against Tiktoken and SentencePiece to see how the sausage is actually made).
- **Learnings**: [001_tokenizer_behaviour.md](./learnings/lab-001/001_tokenizer_behaviour.md)
-  **Results**: [experiment-001.json](./results/lab-001-tokenizer/experiment-001.json)

*More labs coming as I slowly rebuild the transformer architecture block by block...*
