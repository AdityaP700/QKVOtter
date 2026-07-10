# QKV-Otter

(the name of the project sounds cute though)

I'm building this from scratch because I actually want to understand how a Transformer works, rather than just pretending I do by throwing JSON at an OpenAI endpoint.

I throw around buzzwords like "Self-Attention," "KV Cache," and "Context Windows" at parties to sound smart, but under the hood, I'm really just wrestling with the same core mathematical problems that dictate how machines parse human language:

How do I translate messy, infinite human language into a strictly optimized sequence of numbers that a GPU won't choke on?

How does a model figure out what a word means based purely on the geometry of a high-dimensional vector space?

How does a system learn grammar, reasoning, and logic strictly as a byproduct of trying to guess the next word in a sentence?

I don't want to just `pip install transformers` and let HuggingFace sweep these problems under the rug. I want to solve them manually, painfully, and step-by-step.

## The Journey So Far (The Messy Labs)

Here is what I've figured out so far while banging my head against the keyboard:

**Lab 001: The Tokenizer Problem**
How do tokenizer design choices influence what ultimately reaches the transformer? (I compared a custom, naïve BPE against Tiktoken and SentencePiece to see how the sausage is actually made).

- **Experiment 1: Basic Tokenizer Behavior**
  *How do different tokenization algorithms compress the exact same sentence, and what does that mean for the transformer's attention cost?*
  - **Learnings**: [001_tokenizer_behaviour.md](./learnings/lab-001/001_tokenizer_behaviour.md)
  - **Results**: [experiment-001.json](./results/lab-001-tokenizer/experiment-001.json)

- **Experiment 2: Breaking the Tokenizer (Edge Cases)**
  *I wanted to intentionally break the tokenizers by throwing C++ code, SQL queries, Emojis, and non-Latin scripts (Hindi and Japanese) at them to see where they shatter into raw bytes.*
  - **Learnings**: [001_break_tokenizer.md](./learnings/lab-001/001_break_tokenizer.md)
  - **Results**: [experiment-002.json](./results/lab-001-tokenizer/experiment-002.json)

### Key Observation from Lab 001

Instead of simply concluding "tiktoken is better,":

> **Different tokenizers exhibit strengths that largely reflect the corpus on which their vocabularies were learned.**

Examples from this experiment:
1. **tiktoken** compressed source code extremely well.
2. A **toy BPE** struggled with multilingual text because it had never observed those byte patterns frequently.
3. The **SentencePiece** model used in this experiment fragmented code heavily, illustrating that tokenizer quality depends more on the learned vocabulary than on the underlying algorithm itself.

Hence,
- Algorithm $\neq$ Tokenizer quality.
- Vocabulary matters more.

so now the next question is that
"How does tokenizer design affect the transformer that comes afterward?"

**Lab 002: Embeddings**
The embedding layer performs no computation beyond a lookup. Each token ID acts as an index into a trainable embedding matrix, and the corresponding row is returned as that token's vector representation.

- **Experiment 1: Embedding Lookup**
  *Given token IDs produced by a tokenizer, how does the embedding layer convert them into vectors?*
  - **Learnings**: [overall_idea.md](./learnings/lab_002_embeddings/overall_idea.md)
  - **Results ("I love dogs")**: [experiment_001.json](./results/lab_002_embeddings/experiment_001.json)
  - **Results ("I love AI")**: [experiment_002.json](./results/lab_002_embeddings/experiment_001_b.json)

*More labs coming as I slowly rebuild the transformer architecture block by block...*
