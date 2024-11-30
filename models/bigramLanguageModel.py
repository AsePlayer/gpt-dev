import torch
import torch.nn as nn
from torch.nn import functional as F

# This makes sure that random numbers are always the same every time we run the program.
torch.manual_seed(1337)

# This class represents the Bigram Language Model, which learns relationships between pairs of words (bigrams).
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()  # This initializes the base class (nn.Module).
        # This is like a "table" where each word (or token) is mapped to scores for the next possible words.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx: The input tokens (like the words in a sentence). Shape is (batch_size, sequence_length).
        targets: The expected next words (used to calculate how wrong the predictions are). Optional.
        """
        # Use the input tokens to look up their "scores" for the next words in the table.
        logits = self.token_embedding_table(idx)  # Shape: (batch_size, sequence_length, vocab_size)

        # If we are not given any target words (to compare to), we don’t calculate the loss.
        if targets is None:
            loss = None
        else:
            # Get the batch size (B), sequence length (T), and vocab size (C) from the logits.
            B, T, C = logits.shape
            # Reshape the logits (predicted scores) and targets (true next words) to flat lists.
            logits = logits.view(B * T, C)  # Make it 2D: one row per word, one column per possible next word.
            targets = targets.view(B * T)  # Make the targets a simple list of correct next words.
            # Calculate how wrong the predictions are compared to the actual targets.
            loss = F.cross_entropy(logits, targets)

        # Return the predicted scores (logits) and the loss if targets were provided.
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        idx: The starting tokens (e.g., a single word to begin the sequence).
        max_new_tokens: The number of new words we want to generate.
        """
        for _ in range(max_new_tokens):
            # Use the model to predict the next word based on the current sequence.
            logits, _ = self(idx)  # We only care about the logits (predicted scores).
            # Take the scores for just the last word in the sequence.
            logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
            # Turn the scores into probabilities (e.g., "word A: 30%, word B: 70%").
            probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, vocab_size)
            # Randomly pick the next word based on the probabilities.
            idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (batch_size, 1)
            # Add the chosen word to the sequence.
            idx = torch.cat((idx, idx_next), dim=1)  # Shape: (batch_size, sequence_length + 1)
        # Return the full sequence (original + new words).
        return idx
