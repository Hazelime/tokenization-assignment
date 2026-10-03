# Recommended hyperparameters from the instructions:
n_layers = 2        # Number of Transformer layers
C = 256             # Hidden dimension
n_heads = 4         # Number of attention heads
ff_dim = 1024       # Feed-forward dimension
T = 256             # Context-length

# My own choices:
dropout_rate = 0.1  # The instructions recommended dropout, but never specified how much.'
epochs = 8
B = 32              # Batch size. I tried 64 but then the GPU ran out of memory.
lr = 0.001          # Learning rate