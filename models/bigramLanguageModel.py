import torch
import torch.nn as nn
from data import hyperparameters as h
from torch.nn import functional as F

# This makes sure that random numbers are always the same every time we run the program.
torch.manual_seed(1337)


# This class represents the Bigram Language Model, which learns relationships between pairs of words (bigrams).
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()  # This initializes the base class (nn.Module).
        # This is like a "table" where each word (or token) is mapped to scores for the next possible words.
        self.token_embedding_table = nn.Embedding(h.vocab_size, h.n_embd)
        self.position_embedding_table = nn.Embedding(h.block_size, h.n_embd)

        self.blocks = nn.Sequential(*[Block(h.n_embd, n_head=h.n_head) for _ in range(h.n_layer)])
        self.ln_f = nn.LayerNorm(h.n_embd)  # final layer norm
        self.lm_head = nn.Linear(h.n_embd, h.vocab_size)

    def forward(self, idx, targets=None):
        """
        idx: The input tokens (like the words in a sentence). Shape is (batch_size, sequence_length).
        targets: The expected next words (used to calculate how wrong the predictions are). Optional.
        """
        B, T = idx.shape

        # Use the input tokens to look up their "scores" for the next words in the table.
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=h.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B,T,C)

        logits = self.lm_head(x)  # Shape: (batch_size, sequence_length, vocab_size)

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
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -h.block_size:]
            # Use the model to predict the next word based on the current sequence.
            logits, loss = self(idx_cond)  # We only care about the logits (predicted scores).
            # Take the scores for just the last word in the sequence.
            logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size), becomes (B, C)
            # Turn the scores into probabilities (e.g., "word A: 30%, word B: 70%").
            probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, vocab_size)
            # Randomly pick the next word based on the probabilities.
            idx_next = torch.multinomial(probs, num_samples=1)  # Shape: (batch_size, 1)
            # Add the chosen word to the sequence.
            idx = torch.cat((idx, idx_next), dim=1)  # Shape: (batch_size, sequence_length + 1)
        # Return the full sequence (original + new words).
        return idx


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(h.n_embd, head_size, bias=False)
        self.query = nn.Linear(h.n_embd, head_size, bias=False)
        self.value = nn.Linear(h.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril((torch.ones(h.block_size, h.block_size))))
        self.head_size = head_size

        self.dropout = nn.Dropout(h.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        # Computer attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Don't communicate with the past
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(h.n_embd, h.n_embd)
        self.dropout = nn.Dropout(h.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * h.n_embd, h.n_embd),  # Projection layer going back into the residual pathway
            nn.Dropout(h.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # Applies layer normalization over a mini-batch of inputs
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x