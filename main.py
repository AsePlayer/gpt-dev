import torch
from models.bigramLanguageModel import BigramLanguageModel

'''Step 1: Read the data'''
# Open the file 'input.txt' and read its content into a variable called `text`.
# This is our dataset, which we will use to train the model.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Print the length of the dataset (number of characters in the file).
print("length of dataset in characters: ", len(text))

# Print the first 1000 characters of the dataset so we can see what it looks like.
print(text[:1000])

'''Step 2: Prepare the data'''
# Get all unique characters from the dataset (like 'a', 'b', etc., but only once).
chars = sorted(list(set(text)))

# Count how many unique characters we have in the dataset.
vocab_size = len(chars)

# Print all the unique characters and how many there are.
print(''.join(chars))
print(vocab_size)

'''Step 3: Map characters to numbers'''
# Create a mapping from characters to numbers (for example, 'a' -> 0, 'b' -> 1, etc.).
stoi = {ch: i for i, ch in enumerate(chars)}  # stoi = string to index
itos = {i: ch for i, ch in enumerate(chars)}  # itos = index to string

# Create functions to encode strings as numbers and decode numbers back into strings.
encode = lambda s: [stoi[c] for c in s]  # Turns "hello" into [7, 4, 11, 11, 14]
decode = lambda l: ''.join([itos[i] for i in l])  # Turns [7, 4, 11, 11, 14] back into "hello"

# Test the encode and decode functions.
print(encode("hii there"))  # Should print numbers
print(decode(encode("hii there")))  # Should print "hii there"

'''Step 4: Convert the entire dataset into numbers'''
# Encode the whole text as a list of numbers, then turn it into a PyTorch tensor.
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)  # Print the shape (size) of the dataset and its data type.
print(data[:1000])  # Print the first 1000 numbers of the encoded dataset.

'''Step 5: Split the dataset'''
# Split the data into training and validation sets (90% training, 10% validation).
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Define the maximum context length (how much text the model sees before making a prediction).
block_size = 8

# Take the first `block_size + 1` characters from the training data to show an example.
train_data[:block_size + 1]

'''Step 6: Look at the training data'''
# Take the first `block_size` characters as input (`x`) and the next `block_size` as targets (`y`).
x = train_data[:block_size]
y = train_data[1:block_size + 1]

# Print what the model sees (context) and what it's trying to predict (target).
for t in range(block_size):
    context = x[:t + 1]  # The model sees this much of the input so far.
    target = y[t]  # This is the next character it should predict.
    print(f"When input is {context} the target is: {target}")

'''Step 7: Batch creation'''
# A batch is a group of examples processed at the same time. We'll create small batches.
torch.manual_seed(1337)  # Set random seed for consistent results.
batch_size = 4  # How many sequences we process at once.
block_size = 8  # Maximum length of input for each sequence.


# Function to generate a batch of data.
def get_batch(split):
    # Choose the data: training or validation.
    data = train_data if split == 'train' else val_data
    # Randomly pick starting points for `batch_size` sequences.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create inputs (x) and targets (y) from those starting points.
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# Generate a batch of training data.
xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)  # Batch of 4 sequences, each of length 8.
print(xb)  # Input data.
print('targets:')
print(yb.shape)  # Corresponding targets.
print(yb)  # Target data.
print('----')

# Print the context and target for each input sequence in the batch.
for b in range(batch_size):  # Loop over each sequence in the batch.
    for t in range(block_size):  # Loop over each time step in the sequence.
        context = xb[b, :t + 1]  # What the model has seen so far.
        target = yb[b, t]  # The next character it needs to predict.
        print(f"When input is {context.tolist()} the target is: {target}")

print('----\n')

'''Step 8: Use the model'''
# Create a Bigram Language Model with the vocabulary size.
m = BigramLanguageModel(vocab_size)

# Run the model on a batch of inputs and targets to get predictions and loss.
logits, loss = m(xb, yb)
print(logits.shape)  # Predicted scores for each input (batch_size * block_size, vocab_size).
print(loss)  # How wrong the predictions were.

'''Step 9: Generate text'''
# Start with a single token (word 0) and generate 100 new tokens.
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
