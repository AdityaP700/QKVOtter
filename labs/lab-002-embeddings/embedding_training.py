import numpy as np

class Vocabulary:
    """Handles vocabulary creation, token-to-index mapping, and lookup operations."""

    def __init__(self, vocab_list):
        """
        Initialize vocabulary with a list of words.

        Args:
            vocab_list: List of words in the vocabulary
        """
        self.vocab = vocab_list
        # Create a lowercase-to-index mapping
        # for robust lookup
        # this exact word.lower() sets up the key-value pair
        # enumerate loops through the vocab
        # gives the index position and value itself
        # used in text preprocessing workflow :
        # text preprocessing workflows
        self.vocab_dict = {word.lower(): i for i, word in enumerate(self.vocab)}

        # vocab ka size le raha hai
        self.vocab_size = len(self.vocab)

    def get_token_id(self, word):
        """Get the token ID for a given word (case-insensitive)."""
        return self.vocab_dict.get(word.lower())

    def get_word(self, token_id):
        """Get the word for a given token ID."""
        return self.vocab[token_id]

    def encode_sequence(self, words):
        """Convert a list of words to their token IDs."""
        return [self.vocab_dict[word.lower()] for word in words]


class EmbeddingLayer:
    """Handles embedding matrix initialization and lookup operations."""

    def __init__(self, vocab_size, embedding_dim, seed=42):
        """
        Initialize the embedding layer.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Initialize the embedding matrix
        # Using a fixed seed for reproducible results during experimentation
        np.random.seed(seed)
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim)

    def lookup(self, token_ids):
        """
        Lookup embeddings for given token IDs.

        Args:
            token_ids: List of token IDs to lookup

        Returns:
            Array of embeddings corresponding to the token IDs
        """
        # This grabs the rows from the embedding matrix corresponding to our token IDs
        return self.embedding_matrix[token_ids]


class SentenceEmbedder:
    """Handles sentence-level embedding operations including averaging."""

    def __init__(self, embedding_layer):
        """
        Initialize with an embedding layer.

        Args:
            embedding_layer: Instance of EmbeddingLayer
        """
        self.embedding_layer = embedding_layer

    def embed_sentence(self, sentence):
        """
        Convert a sentence to its average embedding representation.

        Args:
            sentence: Input sentence as a string

        Returns:
            Tuple of (words, token_ids, input_embeddings, average_embedding)
        """
        # Process the input sentence
        words = sentence.split()
        token_ids = self.embedding_layer.lookup._vocab.encode_sequence(words) if hasattr(self, '_vocab') else None

        # This needs the vocabulary reference
        # We'll fix this in the Pipeline class
        return words

    def compute_average_embedding(self, input_embeddings):
        """
        Compute the average embedding from input embeddings.

        Args:
            input_embeddings: Array of embeddings for each token

        Returns:
            Average embedding vector
        """
        # Compute the average embedding
        # We average across the sequence length (axis=0)
        #  to get a single vector representing the whole sentence
        # we are calculating downward
        average_embedding = np.mean(input_embeddings, axis=0)
        return average_embedding


class LinearLayer:
    """Linear transformation layer for computing logits."""

    def __init__(self, embedding_dim, vocab_size):
        """
        Initialize the linear layer with random weights.

        Args:
            embedding_dim: Input dimension (embedding dimension)
            vocab_size: Output dimension (vocabulary size)
        """
        # so right now we are simply initializing the
        # the weights, the weights are basically
        # we have embedded dim, then taking the vocab_size
        self.W = np.random.randn(embedding_dim, vocab_size)

    def forward(self, x):
        """
        Compute logits from input embedding.

        Args:
            x: Input embedding vector

        Returns:
            Logits vector
        """
        # the logits are calculated using the analogy
        # z=xw+b
        # X is the average embedding value,
        # the value was (4*1) @(4*20) ->(20,1)
        # W shape: (4, 20)
        # (4,) @ (4, 20) -> (20,)

        logits = np.dot(x, self.W)
        return logits


class SoftmaxLayer:
    """Softmax activation for converting logits to probabilities."""

    def compute_probabilities(self, logits):
        """
        Compute probability distribution using softmax.

        Args:
            logits: Raw logits from linear layer

        Returns:
            Probability distribution over vocabulary
        """
        # implementing the softmax function to compute the
        # probability distribution

        # we subtract the maximum value (np.max) from the logits before exponentiation.
        #  This is a crucial engineering trick called numerical stability—
        # it prevents your computer from crashing due
        # to memory overflow if your logits are
        # very large numbers.
        stable_logits = logits - np.max(logits)
        exp_logits = np.exp(stable_logits)

        probabilities = exp_logits / np.sum(exp_logits)
        return probabilities


class LossFunction:
    """Handles loss computation including cross-entropy."""

    def __init__(self, epsilon=1e-15):
        """
        Initialize loss function.

        Args:
            epsilon: Small value for numerical stability in log
        """
        self.epsilon = epsilon

    def create_one_hot(self, vocab_size, target_id):
        """
        Create one-hot encoded target vector.

        Args:
            vocab_size: Size of vocabulary
            target_id: Index of target word

        Returns:
            One-hot encoded vector
        """
        # we are using the CCE
        # reason is to prevent computer doing bad maths
        # One-Hot encoding treats every single word as completely
        #  independent and equally far apart from every
        #  other word in geometric space

        # another reason : To match the shape of
        #  probabilities for the Loss Formula

        # i.e. The cross-entropy formula performs an element-by-element
        # multiplication between the absolute truth (\(P\))
        # and  model's predictions

        # exactly one element is "hot" (set to 1.0)
        # and all other elements are "cold" (set to 0.0).

        p_true = np.zeros(vocab_size)
        p_true[target_id] = 1.0
        return p_true

    def cross_entropy_loss(self, p_true, q_pred):
        """
        Compute cross-entropy loss.

        Args:
            p_true: True probability distribution (one-hot)
            q_pred: Predicted probability distribution

        Returns:
            Cross-entropy loss value
        """
        # predicted probabilities q_pred before
        # now the case is we are cliping the probabilities
        # lets suppose if the q_pred is lesser than epsilion then it will take
        # the value of epsilion else
        # if its quite larger then it will take the value of 1
        #if its between epsillion and 1 then it will stay as it is

        q_pred_clipped = np.clip(q_pred, self.epsilon, 1.0)
        cross_entropy_loss = -np.sum(p_true * np.log(q_pred_clipped))
        return cross_entropy_loss


class TokenPredictor:
    """Handles token prediction from logits/probabilities."""

    def predict_token(self, logits, vocabulary):
        """
        Predict the token with highest logit.

        Args:
            logits: Model output logits
            vocabulary: Vocabulary instance for lookup

        Returns:
            Tuple of (predicted_token_id, predicted_token)
        """
        # The token with the highest logit is the model's current prediction
        predicted_token_id = np.argmax(logits)
        predicted_token = vocabulary.get_word(predicted_token_id)
        return predicted_token_id, predicted_token

class Trainer:
    def __init__(self,pipeline,optimizer):
        self.pipeline=pipeline
        self.optimizer=optimizer
        self.vocab_size=pipeline.vocabulary.vocab_size

    #first step to backpropagate from loss to logits
    def compute_gradient_wrt_logits(self,probabilities ,target_token_id):
        #initialize for the target_token
        one_hot_target=np.zeros(self.vocab_size)
        one_hot_target[target_token_id]=1.0
        #calculate the gradient for the logits (p-y)
        dl_dlogits = probabilities-one_hot_target

        return dl_dlogits

    #orchestration layer
    def backpropagate(self,stored_forward_state,target_token):
        target_token_id=self.pipeline.vocabulary.get_token_id(target_token.lower())
        probabilities = stored_forward_state['probabilities']
        vocab_size=self.pipeline.vocabulary.vocab_size

        dl_dlogits=self.compute_gradient_wrt_logits(
            probabilities,
            target_token_id,
            vocab_size
        )

class EmbeddingPipeline:
    """Main pipeline orchestrating the entire embedding to prediction flow."""

    def __init__(self, vocab_list, embedding_dim=4):
        """
        Initialize the complete pipeline.

        Args:
            vocab_list: List of vocabulary words
            embedding_dim: Dimension of embeddings
        """
        # Initialize all components
        self.vocabulary = Vocabulary(vocab_list)
        self.embedding_layer = EmbeddingLayer(self.vocabulary.vocab_size, embedding_dim)
        self.linear_layer = LinearLayer(embedding_dim, self.vocabulary.vocab_size)
        self.softmax_layer = SoftmaxLayer()
        self.loss_function = LossFunction()
        self.token_predictor = TokenPredictor()

        self.embedding_dim = embedding_dim

    def process_sentence(self, sentence):
        """
        Process a sentence through the entire pipeline.

        Args:
            sentence: Input sentence string

        Returns:
            Dictionary containing all intermediate results
        """
        # Process the input sentence
        words = sentence.split()
        token_ids = self.vocabulary.encode_sequence(words)

        # Lookup embeddings for the input sentence
        input_embeddings = self.embedding_layer.lookup(token_ids)

        # Compute the average embedding
        average_embedding = np.mean(input_embeddings, axis=0)

        # Compute logits through linear layer
        logits = self.linear_layer.forward(average_embedding)

        # Compute probabilities through softmax
        probabilities = self.softmax_layer.compute_probabilities(logits)

        # Predict token
        predicted_token_id, predicted_token = self.token_predictor.predict_token(
            logits, self.vocabulary
        )

        return {
            'words': words,
            'token_ids': token_ids,
            'input_embeddings': input_embeddings,
            'average_embedding': average_embedding,
            'logits': logits,
            'probabilities': probabilities,
            'predicted_token_id': predicted_token_id,
            'predicted_token': predicted_token
        }

    def compute_loss(self, probabilities, true_target):
        """
        Compute cross-entropy loss for a true target.

        Args:
            probabilities: Predicted probability distribution
            true_target: True target word

        Returns:
            Tuple of (target_token_id, loss_value)
        """
        # define a true target
        # get the index of the true target word from the vocab
        target_token_id = self.vocabulary.get_token_id(true_target)

        # Create one-hot encoding
        p_true = self.loss_function.create_one_hot(
            self.vocabulary.vocab_size, target_token_id
        )

        # Compute loss
        loss = self.loss_function.cross_entropy_loss(p_true, probabilities)

        return target_token_id, loss


class TrainingDataLoader:
    """Handles loading and managing training data."""

    def __init__(self, training_data):
        """
        Initialize with training data.

        Args:
            training_data: List of (input_tokens, target_token) tuples
        """
        self.training_data = training_data

    def get_batch(self):
        """Get all training examples."""
        return self.training_data

    def get_sample(self, index):
        """Get a specific training example by index."""
        return self.training_data[index]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Define vocabulary
    vocab = [
        "I", "you", "love", "like", "hate", "AI", "ML", "Python", "Rust", "C++",
        "cats", "dogs", "birds", "apple", "banana", "orange", "India", "Japan",
        "Paris", "London", "what", "?", "is", "japan?",
        "country", "fruit", "programming", "city", "milk", "bones", "seeds"
    ]

    # Define training data
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

    # Initialize pipeline
    pipeline = EmbeddingPipeline(vocab, embedding_dim=4)

    # Initialize data loader
    data_loader = TrainingDataLoader(training_data)

    # Test input sentence
    input_sentence = "what is Japan?"

    # Process the sentence through the pipeline
    results = pipeline.process_sentence(input_sentence)

    # Display results
    print(f"Input Sentence: {input_sentence}")
    print(f"Words: {results['words']}")
    print(f"Token IDs: {results['token_ids']}")
    print(f"\nInput Embeddings (Shape: {results['input_embeddings'].shape}):")
    print(results['input_embeddings'])
    print(f"\nAverage Embedding (Shape: {results['average_embedding'].shape}):")
    print(results['average_embedding'])

    # Display linear layer information
    print(f"\nLinear Weight W (Shape: {pipeline.linear_layer.W.shape}):")
    print(f"[{pipeline.linear_layer.W[0, :3]} ...]")  # Just printing a snippet to avoid cluttering

    print(f"\nLogits (Shape: {results['logits'].shape}):")
    print(results['logits'])

    # Display predictions
    print(f"\nPredicted Token: '{results['predicted_token']}' "
          f"(ID: {results['predicted_token_id']}, "
          f"Logit: {results['logits'][results['predicted_token_id']]:.4f})")

    print("\nProbabilities: ", results['probabilities'])
    print("Sum:", np.sum(results['probabilities']))

    # Compute loss
    true_target = "country"
    target_token_id, loss = pipeline.compute_loss(results['probabilities'], true_target)

    print("\n--- Cross-Entropy Loss Calculation ---")
    print(f"True Target Word: '{true_target}' (Token ID: {target_token_id})")
    print(f"Model's Assigned Probability to '{true_target}': "
          f"{results['probabilities'][target_token_id]:.6f}")
    print(f"Calculated Cross-Entropy Loss: {loss:.6f}")