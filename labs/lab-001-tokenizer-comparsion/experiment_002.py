import os
import json
from experiment_001 import RegexTokenizer, TiktokenTokenizer, SentencePieceTokenizer
#breaking the tokenizer 
if __name__ == "__main__":
    test_cases = [
        "I use std::unordered_map<int, vector<string>>",
        "SELECT * FROM users;",
        "printf(\"%d\", x);",
        "🐍🔥🚀😊",
        "नमस्ते दुनिया",
        "こんにちは世界"
    ]

    print("Training custom BPE...")
    tok = RegexTokenizer()
    # Train the custom BPE on all strings combined to give it *some* chance
    combined_text = "\n".join(test_cases)
    vocab_size = 256 + 50 # 50 merges
    tok.train(combined_text, vocab_size, verbose=False)

    print("Initializing Tiktoken...")
    tik_tok = TiktokenTokenizer()

    print("Training SentencePiece...")
    spm_tok = SentencePieceTokenizer(vocab_size=400)
    spm_tok.train(combined_text)

    all_results = {}

    for i, input_text in enumerate(test_cases):
        print(f"\nProcessing: {input_text.encode('unicode_escape').decode('ascii')}")

        # Custom BPE
        my_bpe_ids = tok.encode(input_text)
        my_bpe_pieces = [tok.decode([idx]) for idx in my_bpe_ids]

        # Tiktoken
        tik_res = tik_tok.process(input_text)

        # SentencePiece
        spm_res = spm_tok.process(input_text)

        all_results[f"test_{i}"] = {
            "input_text": input_text,
            "my_bpe": {
                "ids": my_bpe_ids,
                "pieces": my_bpe_pieces,
                "count": len(my_bpe_ids)
            },
            "tiktoken": tik_res,
            "sentencepiece": {
                "ids": spm_res["ids"],
                "pieces": [p.encode('ascii', 'replace').decode('ascii') for p in spm_res["pieces"]],
                "count": spm_res["count"]
            }
        }

    # Write results
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab-001-tokenizer"))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "experiment-002.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\nResults stored in {output_file}")
