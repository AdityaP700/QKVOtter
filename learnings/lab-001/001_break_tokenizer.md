# Lab 001: Tokenizer Behavior
## Experiment 2: Breaking the Tokenizer

**Raw Results:** [experiment-002.json](../../results/lab-001-tokenizer/experiment-002.json)
**Source Code:** [experiment_002.py](../../labs/lab-001-tokenizer-comparsion/experiment_002.py)

To really understand the limits, I intentionally tried to break the tokenizers with edge cases: C++ code, SQL queries, C-style formats, Emojis, and non-Latin scripts (Hindi and Japanese).

### Experiment 1 - C++ code
`std::unordered_map<int, vector<string>>`

**Results**

| Tokenizer | Tokens |
| --- | --- |
| My BPE | 11 |
| tiktoken | 11 |
| SentencePiece | 34 |

This surprised me. SentencePiece absolutely exploded.

Why? Because it didn't know `unordered`, `_map`, `vector`, `<string>`. Meanwhile both my BPE and tiktoken had learned them. That's actually a nice observation. The correct conclusion is: This particular SentencePiece model has a vocabulary that is not well suited for these inputs.

### Experiment 2 - SQL
`SELECT * FROM users;`

**Results**

| Tokenizer | Tokens |
| --- | --- |
| My BPE | 5 |
| tiktoken | 5 |
| SentencePiece | 14 |

Again, SentencePiece decomposed `SELECT` $\rightarrow$ `S` `EL` `E` `CT` instead of `SELECT`. Interesting.

### Experiment 3 - C printf
`printf("%d", x);`

This one is beautiful.
My tokenizer: `printf` $\rightarrow$ `p` `r` `int` `f`
tiktoken: `printf` (Literally one token.)

That's exactly because GPT tokenizers were trained heavily on GitHub code.

### Experiment 4 - Emojis
`🐍🔥🚀😊`

This one is funny.
My tokenizer: 12 tokens
tiktoken: 11 tokens
SentencePiece: 5

At first glance people would conclude SentencePiece wins. Not necessarily. Those `?` are suspicious. It means the decoder couldn't reconstruct what the pieces actually represent. So I'd verify whether my SentencePiece vocabulary actually contains those emoji pieces or whether they're falling back to unknown tokens.

### Experiment 5 - Hindi
`नमस्ते दुनिया`

Now this is the most interesting one.
My tokenizer: 23 tokens (because it's byte-level. Every UTF-8 byte is becoming separate pieces.)
tiktoken: 13 tokens
SentencePiece: 14 tokens

This tells us something. GPT's tokenizer has actually learned many Hindi byte sequences. Not perfectly. But much better than a toy BPE.

### Experiment 6 - Japanese
`こんにちは世界`

This shocked me.
My tokenizer: 17 tokens
tiktoken: 4 tokens

WOW. That means `こんにちは` exists almost directly inside GPT's vocabulary. Exactly what you'd expect after training on trillions of multilingual tokens. This is probably the biggest insight of the entire lab.

Notice something. Initially my hypothesis was: *tiktoken is good for English.*
My experiment disproves that. Look:
- Japanese: 4 tokens
- Hindi: 13 tokens
- English: 5-11 tokens

So GPT's tokenizer is not merely English-specific. It's simply trained on a much larger and more diverse corpus. That's why experiments matter.

---

### The Cost Perspective
Let's connect this directly to Transformers by comparing the worst-case (Hindi text) attention costs:

| Tokenizer | Tokens | Relative Attention Cost (n²) |
| --- | --- | --- |
| My BPE | 23 | 529 |
| SentencePiece | 14 | 196 |
| tiktoken | 13 | 169 |

hence the next question is that ,how does one token attend to other token
