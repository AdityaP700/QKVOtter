How do tokenizer design choices influence what ultimately reaches the transformer?
We care about

Tokenizer
↓
engineering_questionsmdToken Count
↓
Attention Cost
↓
Context Usage
↓
Inference

so for the experiement 1
our main goal is to run the input text through
-my BPE
- tiktoken
- sentencepiece

Record :
- token IDs
- token Pieces
- token count