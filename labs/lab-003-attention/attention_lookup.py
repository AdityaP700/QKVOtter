import numpy as np
import json
import os

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

sentence = "I love dogs"
words = sentence.split()
token_ids = [vocab[word] for word in words]
embeddings = embedding_matrix[token_ids] # shape [3, 4]

print("Sentence:", sentence)
print("Token IDs:", token_ids)
print("Embeddings shape:", embeddings.shape)
print("="*50)

def attention_context(target_word, scores):
    scores = np.array(scores)
    weights = scores / scores.sum() # normalize to sum=1

    context = np.sum(embeddings * weights[:, None], axis=0) # weighted sum

    return {
        "target_word": target_word,
        "raw_scores": scores.tolist(),
        "attention_weights": weights.tolist(),
        "context_vector": context.tolist()
    }

# 1. Pretend we are encoding "I"
result_I = attention_context("I", [0.6, 0.3, 0.1]) # pays most attention to itself

# 2. Pretend we are encoding "love"
result_love = attention_context("love", [0.2, 0.5, 0.3]) # you gave this

# 3. Pretend we are encoding "dogs"
result_dogs = attention_context("dogs", [0.1, 0.7, 0.2]) # you gave this

results = {
    "sentence": sentence,
    "token_ids": token_ids,
    "embeddings": embeddings.tolist(),
    "attention_results": [result_I, result_love, result_dogs]
}

# Save to JSON
output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab_003_attention")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "manual_attention.json")

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

# Print nicely
for r in results["attention_results"]:
    print(f"=== Encoding: {r['target_word']} ===")
    print("Raw Scores: ", np.round(r["raw_scores"], 3))
    print("Norm Weights: ", np.round(r["attention_weights"], 3))
    print("Context Vector: ", np.round(r["context_vector"], 4))
    print()

print(f"Results saved to {os.path.relpath(output_file)}")