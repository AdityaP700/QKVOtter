import numpy as np
import json
import os
import argparse

vocab_size = 5
embedding_dim = 4
np.random.seed(42) # for reproducibility
embedding_matrix = np.random.randn(vocab_size, embedding_dim)

vocab = {
    "I": 0,
    "love": 1,
    "AI": 2,
    "cats": 3,
    "dogs": 4
}

parser = argparse.ArgumentParser(description="Run weighted embedding experiment.")
parser.add_argument("--sentence1", type=str, default="I love dogs", help="First input sentence")
parser.add_argument("--sentence2", type=str, default="dogs love I", help="Second input sentence")
parser.add_argument("--output", type=str, default="experiment_002_weighted.json", help="Output JSON filename")
args = parser.parse_args()

def process_sentence(sentence, weights):
    words = sentence.split()
    token_ids = [vocab[word] for word in words] # will crash if word not in vocab

    input_embeddings = embedding_matrix[token_ids] # shape [seq_len, embedding_dim]

    # Hardcoded weights. Must match sentence length
    if len(weights)!= len(token_ids):
        raise ValueError(f"Weights length {len(weights)}!= sentence length {len(token_ids)}")

    weights = np.array(weights)
    # weights[:, None] makes it shape [seq_len, 1] so it broadcasts across embedding_dim
    sentence_embedding = np.sum(input_embeddings * weights[:, None], axis=0)

    return {
        "input_sentence": sentence,
        "token_ids": token_ids,
        "token_embeddings": input_embeddings.tolist(),
        "weights_used": weights.tolist(),
        "weighted_embedding": sentence_embedding.tolist(),
        "shape": list(input_embeddings.shape)
    }

# Example: 3-word sentences, so 3 weights
weights = [0.1, 0.2, 0.7] # gives more importance to last word

result1 = process_sentence(args.sentence1, weights)
result2 = process_sentence(args.sentence2, weights)

output_data = {
    "sentence1": result1,
    "sentence2": result2
}

output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab_002_embeddings")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, args.output)

with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

# Print to console
print("=== SENTENCE 1 ===")
print("Sentence:", result1["input_sentence"])
print("Token IDs:", result1["token_ids"])
print("Weights:", result1["weights_used"])
print("Weighted Embedding:", np.round(result1["weighted_embedding"], 4))

print("\n=== SENTENCE 2 ===")
print("Sentence:", result2["input_sentence"])
print("Token IDs:", result2["token_ids"])
print("Weights:", result2["weights_used"])
print("Weighted Embedding:", np.round(result2["weighted_embedding"], 4))

print(f"\nResults saved to {os.path.relpath(output_file)}")