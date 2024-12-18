import torch

# # hyperparameters
# batch_size = 16  # how many independent sequences will we process in parallel?
# block_size = 32  # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 500
# learning_rate = 1e-3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 64
# vocab_size = 0
# n_head = 4
# n_layer = 4
# dropout = 0.1

# Revised hyperparameters
batch_size = 32  # Increased batch size for faster training
block_size = 64  # Larger context length for improved predictions
max_iters = 3000  # Fewer iterations to cap training time
eval_interval = 300  # More frequent evaluations for better monitoring
learning_rate = 5e-4  # Reduced learning rate for finer updates
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100  # Fewer evaluation iterations to save time
n_embd = 128  # Larger embedding size for better representation
vocab_size = 0  # Placeholder, keep unchanged
n_head = 8  # Increased number of heads for more expressivity
n_layer = 3  # Slightly fewer layers to reduce training time
dropout = 0.2  # Higher dropout for better regularization