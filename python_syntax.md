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

