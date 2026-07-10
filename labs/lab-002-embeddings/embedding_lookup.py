import numpy as np
vocab_size = 5

#i am taking the embedding dimension to be 4 as of now
embedding_dim=4

#Why does every token need one row?
#cuz each token iD acts as the row index inside the
#embedding matrix

embedding_matrix=np.random.randn(vocab_size,embedding_dim)

#the question is how the embedding layer converts the token ids into vectors
#this is our tokenizer's vocabulary
#The tokenizer maps each word in the sentence
#to its corresponding token ID using the vocabulary.
#hence i have to split(mock)
#then i have to print in such that it produces an array
#you have a key-value pair
#you have to carry out the key and the value whenever prompted

vocab = {
    "I":0,
    "love":1,
    "AI":2,
    "cats":3,
    "dogs":4
}
vocab_values = list(vocab.values())
input_sentence="I love AI"
words=input_sentence.split()

# tokens_ids=[]
# for word in words:
#     number = vocab[word]
#     tokens_ids.append(number)

token_ids=[vocab[word]for word in words]
#for every word in words,look into the vocab dict
#and try to map out the things

#this technique known as list compression
#declaring the token input

# input_embeddings=[]
# for token_id in token_ids:
#     row_vector=embedding_matrix[token_id]
#     input_embeddings.append(row_vector)

# input_embeddings=np.array(input_embeddings)

#integer array indexing
#it reads out that ,look at the embedding_matrix ,pull out the specific
# rows corresponding to the index number listed in token_ids
#stack together to form a new matrix called input embeddings
input_embeddings=embedding_matrix[token_ids]
print("Token IDs:", token_ids)
print("\nInput Embeddings:")
print(input_embeddings)
print("Shape:", input_embeddings.shape)