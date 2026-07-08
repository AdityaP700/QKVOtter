# Lab 001: Tokenizer Behavior

## How do tokenizer design choices influence what ultimately reaches the transformer?

When building language models, the very first step in the pipeline is tokenization. I care deeply about tokenizer choices because they have a cascading effect on model performance, efficiency, and intelligence.

The pipeline of consequences looks like this:
**Tokenizer $\rightarrow$ Token Count $\rightarrow$ Attention Cost $\rightarrow$ Context Usage $\rightarrow$ Inference**

- **Token Count**: A less efficient tokenizer splits words into too many small pieces (e.g., characters).
- **Attention Cost**: Attention is an $O(N^2)$ operation. If a tokenizer outputs 20% more tokens, the attention mechanism takes significantly more memory and compute time.
- **Context Usage**: Models have a fixed context window (e.g., 8k tokens). A highly compressed tokenization scheme means you can fit more actual *words* and *concepts* into that 8k window.
- **Inference Speed**: Predicting fewer tokens per word means generating text faster for the end user.

---

## The Core Experiment

**Raw Results:** [experiment-001.json](../../results/lab-001-tokenizer/experiment-001.json)
**Source Code:** [experiment_001.py](../../labs/lab-001-tokenizer-comparsion/experiment_001.py)

**Input Text:** `"I'm building chunkdUp in RUST"`

I ran this string through three different tokenizers to observe how they split the exact same text. Here is what I found, broken down into 8 key observations that make the underlying mechanics crystal clear:

### Observation 1 - Compression Efficiency

| Tokenizer | Tokens |
| --- | --- |
| My BPE | 18 |
| SentencePiece | 12 |
| tiktoken | 9 |

Immediately we can say: **tiktoken compressed this sentence the best.**

Why? Because its vocabulary already contains merges like `" chunk"`, `"Up"`, `" in"`, `" R"`, `"UST"`. Instead of `c h u n k`, it already knows `chunk` because it probably saw it thousands or millions of times during tokenizer training.

### Observation 2 - My tokenizer behaves exactly as expected

This is actually good. Look at how `chunkdUp` became:
`c h u n k d U p`

Why? Not because my implementation is bad. Because my tokenizer has probably trained on very little data. It has never observed `chunk`, `chunkd`, or `chunkdUp` often enough to merge them. Karpathy repeatedly mentions this in minBPE: **BPE only merges what it repeatedly sees.**

### Observation 3 - SentencePiece segmented differently

Look carefully at `building`:
`building` $\rightarrow$ `bu` `il` `ding`

Interesting. It didn't learn `building`. It learned `bu`, `il`, and `ding`. Similarly, `chunkdUp` became `ch`, `un`, `kdUp`.

This tells us something: SentencePiece's learned vocabulary preferred these subword units. This is neither better nor worse. It simply reflects what was statistically useful during tokenizer training.

### Observation 4 - tiktoken is aggressively optimized

This surprised me the most. Look at `RUST`:
`RUST` $\rightarrow$ `R` `UST`

Why not `RU` `ST`? Or `RUST`? Because during tokenizer training, `UST` probably appeared much more often. Think about words like `JUST`, `MUST`, `TRUST`, `DUST`. Therefore, `UST` became an extremely valuable merge. This is vocabulary optimization in action.

### Observation 5 - Spaces are treated differently

This is one of the biggest design differences.
- **My tokenizer:** `building` $\rightarrow$ `" building"`
- **SentencePiece:** `building` $\rightarrow$ `▁building` (represented under the hood as `\u2581bu` `il` `ding`)

The underscore `▁` is literally "beginning of a word". SentencePiece makes whitespace explicit. GPT's tokenizer simply learns `" building"` as another byte sequence. Different philosophy. Same purpose.

### Observation 6 - Vocabulary reflects corpus statistics

This is probably the most important conclusion. Notice `chunkdUp` doesn't exist in any tokenizer. Makes sense. Because it's my project. Nobody trained on it.

But **tiktoken still compresses it surprisingly well:** `chunk` `d` `Up` because `chunk` and `Up` are individually common.

That tells us: **tokenizers don't memorize words. They memorize useful pieces.** Huge difference.

### Observation 7 - This affects transformer cost

Let's quantify it. Suppose attention complexity is $O(n^2)$.

| Tokenizer | Tokens | Relative Attention Cost (n²) |
| --- | --- | --- |
| My BPE | 18 | 324 |
| SentencePiece | 12 | 144 |
| tiktoken | 9 | 81 |


So, for exactly the same sentence, GPT performs roughly `81` instead of `324` pairwise attention computations. That's a **4× reduction**. Of course, this is a tiny example, but it illustrates the direction and massive downstream impact of tokenization on compute costs.

