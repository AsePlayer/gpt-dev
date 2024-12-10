import torch
from torch.nn import functional as F
from models.bigramLanguageModel import BigramLanguageModel

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

'''Read the data'''
# Open the file 'input.txt' and read its content into a variable called `text`.
# This is our dataset, which we will use to train the model.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

'''Prepare the data'''
# Get all unique characters from the dataset (like 'a', 'b', etc., but only once).
chars = sorted(list(set(text)))

# Count how many unique characters we have in the dataset.
vocab_size = len(chars)

'''Map characters to numbers'''
# Create a mapping from characters to numbers (for example, 'a' -> 0, 'b' -> 1, etc.).
stoi = {ch: i for i, ch in enumerate(chars)}  # stoi = string to index
itos = {i: ch for i, ch in enumerate(chars)}  # itos = index to string

# Create functions to encode strings as numbers and decode numbers back into strings.
encode = lambda s: [stoi[c] for c in s]  # Turns "hello" into [7, 4, 11, 11, 14]
decode = lambda l: ''.join([itos[i] for i in l])  # Turns [7, 4, 11, 11, 14] back into "hello"

'''Train and test splits'''
# Encode the whole text as a list of numbers, then turn it into a PyTorch tensor.
data = torch.tensor(encode(text), dtype=torch.long)

# Split the data into training and validation sets (90% training, 10% validation).
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data loading
# Function to generate a batch of data.
def get_batch(split):
    # Choose the data: training or validation.
    data = train_data if split == 'train' else val_data
    # Randomly pick starting points for `batch_size` sequences.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create inputs (x) and targets (y) from those starting points.
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # No back propagation
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Take the first `block_size + 1` characters from the training data to show an example.
train_data[:block_size + 1]

'''Look at the training data'''
# Take the first `block_size` characters as input (`x`) and the next `block_size` as targets (`y`).
x = train_data[:block_size]
y = train_data[1:block_size + 1]


'''Batch creation'''
# A batch is a group of examples processed at the same time. We'll create small batches.
torch.manual_seed(1337)  # Set random seed for consistent results.
batch_size = 4  # How many sequences we process at once.
block_size = 8  # Maximum length of input for each sequence.

# Generate a batch of training data.
xb, yb = get_batch('train')


'''Use the model'''
# Create a Bigram Language Model with the vocabulary size.
model = BigramLanguageModel(vocab_size)
m = model.to(device)

'''Create Optimizer'''
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
