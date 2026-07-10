import numpy as np
vocab_size = 5
#i am taking the embedding dimension to be 4 as of now
embedding_dim=4
#Why does every token need one row?
#cuz each token iD acts as the row index inside the
#embedding matrix
embedding_matrix=np.random.randcn(vocab_size,embedding_dim)

#the question is how the embedding layer converts the token ids into vectors
#this is our tokenizer's vocabulary
#The tokenizer maps each word in the sentence
#to its corresponding token ID using the vocabulary.
#hence i have to split(mock)
#then i have to print in such that it produces an array
vocab = {
    "I":0,
    "love":1,
    "AI":2,
    "cats":3,
    "dogs":4
}

input_sentence="I love AI"
words=input_sentence.split()

#declaring the token input
