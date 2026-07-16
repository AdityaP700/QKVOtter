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

# tokens_ids=[]
# for word in words:
#     number = vocab[word]
#     tokens_ids.append(number)

token_ids=[vocab[word]for word in words]
#for every word in words,look into the vocab dict
#and try to map out the things
#this technique known as list compression


#  Slicing (Basic Indexing)
# Grab rows 0 and 1, and all 4 columns
first_two_embeddings = embedding_matrix[0:2, :]

# Boolean Masking (Boolean Indexing)
# 1. Creates a True/False matrix of the same shape
mask = embedding_matrix > 0.5

# 2. Extracts only the numbers that matched True into a flat list
high_values = embedding_matrix[mask]

# 3. Multi-Dimensional Indexing (Coordinate Indexing)

single_weight = embedding_matrix[0,2]



----experimentation-2
- the case is ,Token-to-ID Mapping (Word Embedding)Before feeding text into a neural network (like an LSTM or Transformer),

- every word must be converted into a unique identification number. This dictionary acts as the master lookup index.

# vocab_dict = {word.lower(): i for i, word in enumerate(vocab)}


another is the np.random.seeed(42);
# If you are trying to invent a new game, and your game crashes, you want to be able to replay the exact same moves to figure out why it broke. If the numbers change every time, you can never recreate the bug!

it is used to reproduce exact same random outputs

# .lower()
- it is a built-in string method used to convert all uppercase letters in a text string into lowercase letters.
