import numpy as np
import json
import os

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

    #logits->weights yayyyy lesgoo
    def compute_gradient_wrt_weights(self, dl_dlogits, average_embedding):
        """
    Compute gradient of loss with respect to linear weights W.
    Forward: logits = avg_embedding @ W
    Backward: dL/dW = avg_embedding^T ⊗ dL/dlogits
    """
        dl_dw=np.outer(average_embedding,dl_dlogits)
        return dl_dw

    def compute_gradient_wrt_average_embedding(self, dl_dlogits, W):
        """
        Compute gradient of loss with respect to average embedding.
    Forward: logits = avg_embedding @ W
    Backward: dL/davg_embedding = dL/dlogits @ W^T
        """
        dl_davg_embedding = W @ dl_dlogits
        return dl_davg_embedding

    def compute_gradient_wrt_embeddings(self, dl_davg_embedding, num_tokens):
         """
    Distribute the average embedding gradient to each individual token.
    Forward: avg = (e_1 + e_2 + ... + e_n) / n
    Backward: dL/de_i = (1/n) * dL/davg

    Args:
        dl_davg_embedding: Gradient from average embedding, shape (embedding_dim,)
        num_tokens: Number of tokens in the input sequence

    Returns:
        dl_dindividual: Gradient for each token's embedding, shape (embedding_dim,)
    """
         dl_dindividual = dl_davg_embedding / num_tokens
         return dl_dindividual

    def compute_gradient_wrt_embedding_matrix(self, dl_dindividual, token_ids):
        """
    Map individual token gradients to the embedding matrix.
    Only rows corresponding to input tokens get gradients.

    Args:
        dl_dindividual: Gradient for each token, shape (embedding_dim,)
        token_ids: List of token IDs from the input

    Returns:
        dl_dE: Sparse gradient matrix, shape (vocab_size, embedding_dim)
              All rows are zero except for input token rows
    """
        dl_dE = np.zeros((self.vocab_size, dl_dindividual.shape[0]))
        for token_id in token_ids:
            dl_dE[token_id]+=dl_dindividual

        return dl_dE


    #orchestration layer
    def backpropagate(self,stored_forward_state,target_token):
        probabilities = stored_forward_state['probabilities']
        average_embedding = stored_forward_state['average_embedding']
        token_ids = stored_forward_state['token_ids']
        num_tokens = stored_forward_state['num_tokens']
        W = self.pipeline.linear_layer.W

        # Step 1: Loss → Logits
        target_token_id = self.pipeline.vocabulary.get_token_id(target_token.lower())
        dl_dlogits = self.compute_gradient_wrt_logits(probabilities, target_token_id)

        # Step 2: FORK - Compute both paths
        dl_dW = self.compute_gradient_wrt_weights(dl_dlogits, average_embedding)
        dl_davg_embedding = self.compute_gradient_wrt_average_embedding(dl_dlogits, W)

        # Step 3: Continue downward to embeddings
        dl_dindividual=self.compute_gradient_wrt_embeddings(dl_davg_embedding,num_tokens)

        # Step 4: Individual → Embedding Matrix (SPARSE ASSIGNMENT)
        dl_dE=self.compute_gradient_wrt_embedding_matrix(dl_dindividual, token_ids)
        gradients = {
        'W': dl_dW,
        'embeddings': dl_dE,
    }

        return gradients

    def train_step(self, input_tokens, target_token):
        # Forward pass
        sentence = ' '.join(input_tokens)
        forward_results = self.pipeline.process_sentence(sentence)

    # Add num_tokens to stored state (the one thing pipeline doesn't store yet)
        forward_results['num_tokens'] = len(input_tokens)

    # Compute loss - USE YOUR EXISTING METHOD!
        target_token_id, loss = self.pipeline.compute_loss(
        forward_results['probabilities'],
        target_token
    )

    # Backward pass
        gradients = self.backpropagate(forward_results, target_token)

    # Update parameters
        self.optimizer.update(self.pipeline, gradients)

        return loss, forward_results['predicted_token']

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
            'predicted_token': predicted_token,
            'num_tokens': len(words)
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

class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, pipeline, gradients):
        # Update linear weights
        pipeline.linear_layer.W -= self.learning_rate * gradients['W']

        # Update embedding matrix
        pipeline.embedding_layer.embedding_matrix -= self.learning_rate * gradients['embeddings']


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


    # ============================================================================
    # TRAINING DEMONSTRATION
    # ============================================================================

    # Re-initialize for a clean training run
    pipeline = EmbeddingPipeline(vocab, embedding_dim=4)
    optimizer = SGDOptimizer(learning_rate=0.01)
    trainer = Trainer(pipeline, optimizer)

    # Test on a single example
    input_tokens = ["what", "is", "japan?"]
    target_token = "country"
    target_tokenId = pipeline.vocabulary.get_token_id(target_token)

    print("\n=== BEFORE TRAINING ===")
    result_before = pipeline.process_sentence("what is Japan?")
    print(f"Predicted: '{result_before['predicted_token']}'")
    print(f"Probabilities for '{target_token}': {result_before['probabilities'][target_tokenId]:.6f}")

    training_history = {
        "forward_pass_initial": {
            "predicted_token": result_before["predicted_token"],
            "target_probability": float(result_before['probabilities'][target_tokenId])
        },
        "epochs": []
    }

    print("\n=== TRAINING 1000 STEPS ===")
    for step in range(1001):
        loss, predicted = trainer.train_step(input_tokens, target_token)

        # Save every step to history or just every 10 steps to save space. Let's save every 10 steps.
        if step % 10 == 0:
            result = pipeline.process_sentence("what is Japan?")
            prob = result['probabilities'][target_tokenId]
            training_history["epochs"].append({
                "step": step,
                "loss": float(loss),
                "target": vocab[target_tokenId],
                "prediction": result['predicted_token'],
                "p_target": float(prob)
            })

        if step % 100 == 0:
            result = pipeline.process_sentence("what is Japan?")
            probabilities = result['probabilities']
            print(f"--- Step {step} ---")
            print("Target      :", vocab[target_tokenId])
            print("Prediction  :", vocab[np.argmax(probabilities)])
            print("P(target)   :", probabilities[target_tokenId])
            print("Loss        :", loss)
            print()

    # Final check
    result_final = pipeline.process_sentence("what is Japan?")
    print("\n=== AFTER TRAINING ===")
    print(f"Final prediction: '{result_final['predicted_token']}'")
    print(f"Final probability for '{target_token}': {result_final['probabilities'][target_tokenId]:.6f}")

    training_history["forward_pass_final"] = {
        "predicted_token": result_final["predicted_token"],
        "target_probability": float(result_final['probabilities'][target_tokenId])
    }

    # Save to file
    results_dir = r"D:\Development\Projects\core\TransFi\results\lab_002_embeddings"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    print(f"\nTraining results saved to: {results_path}")