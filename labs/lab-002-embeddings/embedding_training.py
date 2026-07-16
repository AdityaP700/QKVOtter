import numpy as np

vocab = [
    "I",
    "you",
    "love",
    "like",
    "hate",
    "AI",
    "ML",
    "Python",
    "Rust",
    "C++",
    "cats",
    "dogs",
    "birds",
    "apple",
    "banana",
    "orange",
    "India",
    "Japan",
    "Paris",
    "London"
]

training_data = [
    (["i", "love"], "ai"),
    (["i", "love"], "python"),
    (["i", "love"], "rust"),
    (["you", "love"], "ai"),

    (["cats", "like"], "milk"),
    (["dogs", "like"], "bones"),
    (["birds", "like"], "seeds"),

    (["apple", "is"], "fruit"),
    (["banana", "is"], "fruit"),
    (["orange", "is"], "fruit"),

    (["python", "is"], "programming"),
    (["rust", "is"], "programming"),
    (["c++", "is"], "programming"),

    (["paris", "is"], "city"),
    (["london", "is"], "city"),
    (["tokyo", "is"], "city"),

    (["india", "is"], "country"),
    (["japan", "is"], "country"),
    (["france", "is"], "country")
]

input_sentence = "I love India"

# Create a lowercase-to-index mapping
# for robust lookup
#this exact word.lower() sets up the key-value pair
#enumerate loops through the vocab
#gives the index position and value itself
#used in text preprocessing workflow :
#text preprocessing workflows
vocab_dict = {word.lower(): i for i, word in enumerate(vocab)}

#vocab ka size le raha hai
vocab_size = len(vocab)
embedding_dim = 4

# Initialize the embedding matrix
# Using a fixed seed for reproducible results during experimentation
np.random.seed(42)
embedding_matrix = np.random.randn(vocab_size, embedding_dim)

# Process the input sentence
words = input_sentence.split()
token_ids = [vocab_dict[word.lower()] for word in words]

# Lookup embeddings for the input sentence
# This grabs the rows from the embedding matrix corresponding to our token IDs
input_embeddings = embedding_matrix[token_ids]

# Compute the average embedding
# We average across the sequence length (axis=0)
#  to get a single vector representing the whole sentence
# we are calculating downward
average_embedding=np.mean(input_embeddings,axis=0)
print(f"Input Sentence: {input_sentence}")
print(f"Words: {words}")
print(f"Token IDs: {token_ids}")
print(f"\nInput Embeddings (Shape: {input_embeddings.shape}):")
print(input_embeddings)
print(f"\nAverage Embedding (Shape: {average_embedding.shape}):")
print(average_embedding)
