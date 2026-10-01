from tokenizers import Tokenizer
from build_tokenizers import CharTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from models import TransformerBlock, DecoderTransformer
from pathlib import Path
import matplotlib.pyplot as plt
from plotting import plot_losses, plot_loss_comparison


class LMDataset(Dataset):
    """
    A PyTorch-compatible dataset class taking a list of tokens and a context length (T) as
    input and returning batches of tokens (x) and the same tokens but shifted one context
    length (T) to the right (y) when indexed.

    Parameters:
        tokens: list
            A list of tokens (integers) representing the encoded text data.
        T: int
            The context length for the dataset. Each sample will consist of T tokens. 
    """
    def __init__(self, tokens, T):
        self.tokens = torch.tensor(tokens, dtype = torch.long)
        self.T = T

    def __len__(self):
        # // is whole-number division (i.e. discard remainder).
        return len(self.tokens - 1) // self.T

    def __getitem__(self, i):
        """
        Magic method to enable use of square brackets for class objects, e.g. dataset[12:14].
        """
        start = i * self.T
        # y is just x shifted one context length (T) to the right.
        x = self.tokens[start:start + self.T]
        y = self.tokens[start + 1:start + self.T + 1]
        return x, y


def get_encoded_dataloader(paths: list[str],
                           tokenizer: Tokenizer or CharTokenizer,
                           T: int,
                           B: int) -> DataLoader:
    """
    Encodes the text data from the given file paths using the provided tokenizer and
    returns a DataLoader.

    Parameters:
        paths: list[str]
            A list of file paths containing the text data to be encoded.
        tokenizer: Tokenizer or CharTokenizer
            The tokenizer to use for encoding the text data.
        T: int
            The context length for the dataset.
        B: int
            The batch size for the DataLoader.
    
    Returns:
        DataLoader
            A DataLoader that provides batches of encoded tokens.
    """
    encoded_data = []
    eos_idx = tokenizer.encode("[EOS]").ids
    for path in paths:
        with open(path, "r", encoding = "utf-8") as f:
            for line in f:
                encoded_data.extend(tokenizer.encode(line.strip()).ids)
                # Separate lines with [EOS] tokens.
                encoded_data.extend(eos_idx)
    encoded_dataset = LMDataset(encoded_data, T)
    # Shuffle to prevent order bias.
    return DataLoader(encoded_dataset, B, shuffle = True)


def evaluate(model: DecoderTransformer,
             dataloader: DataLoader,
             loss_function: nn.CrossEntropyLoss,
             device: torch.device) -> float:
    """
    Evaluate the model on a given dataset and compute average cross-entropy loss per token.

    Parameters:
        model: DecoderTransformer
            The DecoderTransformer model to evaluate.
        dataloader: DataLoader
            A DataLoader providing batches of tokens (x) and the same tokens but shifted
            one context length (T) to the right (y).
        loss_function: nn.CrossEntropyLoss
            The loss function to use for computing the loss.
    
    Returns:
        float
            The average loss per token for the given dataset.
    """
    # Set model to evaluation mode.
    model.eval()
    
    # Disable gradient computation since we are only evaluating. Apparently, inference_mode()
    # is more efficient than no_grad().
    with torch.inference_mode():
        total_loss, total_tokens = 0.0, 0
        
        for x, y in dataloader:
            # Only move the current batch to GPU rather than the whole dataset. 
            x = x.to(device)
            y = y.to(device)

            # Flatten the logits and corresponding targets since CrossEntropyLoss expects the class
            # dimension (V) second. We're making B * T predictions either way.
            logits = model(x)
            b, t, v = logits.shape
            loss = loss_function(logits.reshape(b * t, v),
                                 y.reshape(b * t))

            # y.numel() returns the number of elements (tokens in this case) in a tensor.
            n_tokens = y.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    
    # Set model back to training mode.
    model.train()
    
    # Return average loss per token for the given dataset.
    return total_loss / total_tokens


def train_model(tokenizer: Tokenizer or CharTokenizer,
                train_loader: DataLoader,
                valid_loader: DataLoader,
                device: torch.device,
                name: str,
                C: int = 256,
                n_layers: int = 2,
                n_heads: int = 8,
                ff_dim: int = 512,
                dropout_rate: float = 0.1,
                T: int = 1024,
                epochs: int = 5,
                lr: float = 0.001) -> tuple[DecoderTransformer, list, list]:
    """
    Trains a DecoderTransformer model on the given training dataset and evaluates it on the
    validation dataset.

    Parameters:
        tokenizer: Tokenizer or CharTokenizer
            The tokenizer used for encoding the text data.
        train_loader: DataLoader
            A DataLoader providing batches of tokens (x) and the same tokens but shifted
            one context length (T) to the right (y) for training.
        valid_loader: DataLoader
            A DataLoader providing batches of tokens (x) and the same tokens but shifted
            one context length (T) to the right (y) for validation.
        name: str
            The name of the model or experiment, used for logging and plotting.
        C: int
            The hidden dimension of the model.
        n_layers: int
            The number of Transformer layers in the model.
        n_heads: int
            The number of attention heads in the model.
        ff_dim: int
            The feed-forward dimension in the model.
        dropout_rate: float
            The dropout rate used in the model.
        T: int
            The context length for the dataset.
        epochs: int
            The number of epochs to train the model.
        lr: float
            The learning rate for the optimizer.
        device: torch.device
            The device (CPU or GPU) to use for training and evaluation.
    
    Returns:
        tuple[DecoderTransformer, list, list]
            A tuple containing the trained model, a list of training losses for each epoch,
            and a list of validation losses for each epoch.
    """
    print(f"Setting up model #{n}...")
    model = DecoderTransformer(V = len(tokenizer.get_vocab()),
                                C = C,
                                n_layers = n_layers,
                                n_heads = n_heads,
                                ff_dim = ff_dim,
                                dropout_rate = dropout_rate,
                                T = T).to(device)
    # Calculate model size by number of parameters.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model #{n} has {n_params:,} parameters.")
    optimizer = optim.Adam(model.parameters(), lr = lr)
    loss_function = nn.CrossEntropyLoss()

    # Calculate pre-training loss on training and validation datasets.
    train_losses, valid_losses = [], []
    print("Calculating pre-training loss...")
    train_losses.append(evaluate(model, train_loader, loss_function, device))
    valid_losses.append(evaluate(model, valid_loader, loss_function, device))

    # Train
    print("Training model...")
    for epoch in range(epochs):
        i, total_train_loss, total_train_tokens = 0, 0, 0
        for x, y in train_loader:
            i += 1
            # Just a way of showing progress.
            if i % 200 == 0:
                print(f" ↳ Epoch {epoch+1}, batch {i}/{len(train_loader)}")
            
            # Only move the current batch to GPU rather than the entire dataset.
            x = x.to(device)
            y = y.to(device)
            
            # Step 1: Reset gradients.
            optimizer.zero_grad()

            # Step 2: Forward pass.
            logits = model(x)

            # Step 3: Compute loss after reshaping (B, T, V) -> (B * T, V), since 
            # CrossEntropyLoss expects the class variable (V) second. We're making B * T
            # predictions either way.
            b, t, v = logits.shape
            loss = loss_function(logits.reshape(b * t, v), y.reshape(b * t))

            # Gather total loss already here so we don't need to redo it with evaluate().
            # Doing it here means we are doing it while the model is changing, but it
            # should be close enough to the loss after the epoch is done.
            n_tokens = y.numel()
            total_train_loss += loss.item() * n_tokens
            total_train_tokens += n_tokens

            # Step 4: Backward pass.
            loss.backward()

            # Step 5: Update parameters.
            optimizer.step()
    
        # Calculate loss on train and validation datasets after each epoch.
        train_losses.append(total_train_loss / total_train_tokens)
        valid_losses.append(evaluate(model, valid_loader, loss_function, device))

    # Summarize loss on train and validation datasets per epoch.
    print("Average cross-entropy loss per token for training and validation loss.")
    for epoch in range(epochs + 1):
        print(f"Epoch {epoch}: {train_losses[epoch]:.4f} and {valid_losses[epoch]:.4f}")
    
    return model, train_losses, valid_losses


if __name__ == '__main__':
    # Sample syntax:
    # python3 train_models.py 

    # Hyperparameters. Use recommended values and epochs = 5, batch_size = 32, lr = 0.001.
    n_layers = 2        # Number of Transformer layers
    C = 256             # Hidden dimension
    n_heads = 4         # Number of attention heads
    ff_dim = 1024       # Feed-forward dimension
    dropout_rate = 0.1 
    T = 256             # Context-length
    epochs = 5
    B = 32              # Batch size. I tried 64 but then the GPU ran out of memory.
    lr = 0.001          # Learning rate.

    train_paths = ["/srv/data/lt2326-h26/a1/train/en.txt", 
                   "/srv/data/lt2326-h26/a1/train/tr.txt", 
                   "/srv/data/lt2326-h26/a1/train/zh.txt"]
    valid_paths = ["/srv/data/lt2326-h26/a1/valid/en.txt", 
                   "/srv/data/lt2326-h26/a1/valid/tr.txt", 
                   "/srv/data/lt2326-h26/a1/valid/zh.txt"]

    # Load the tokenizers.
    tokenizers = [CharTokenizer().from_file("tokenizers/char_tokenizer.json"), 
                  Tokenizer.from_file("tokenizers/bpe_10000_tokenizer.json"),
                  Tokenizer.from_file("tokenizers/bpe_20000_tokenizer.json")]
    
    # Encode the training and validation data with different tokenizers.
    print("Encoding training and validation data with the character-level tokenizer...")
    char_train = get_encoded_dataloader(train_paths, tokenizers[0], T, B)
    char_valid = get_encoded_dataloader(valid_paths, tokenizers[0], T, B)
    print("Encoding training and validation data with the small BPE tokenizer...")
    small_bpe_train = get_encoded_dataloader(train_paths, tokenizers[1], T, B)
    small_bpe_valid = get_encoded_dataloader(valid_paths, tokenizers[1], T, B)
    print("Encoding training and validation data with the big BPE tokenizer...")
    big_bpe_train = get_encoded_dataloader(train_paths, tokenizers[2], T, B)
    big_bpe_valid = get_encoded_dataloader(valid_paths, tokenizers[2], T, B)
    print("Encoding done.")

    # Make sure to use GPUs if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")

    # Create one model for each tokenizer.
    n, losses = 1, {}
    for tokenizer, train_loader, valid_loader, name in zip(tokenizers,
                                                           [char_train, small_bpe_train, big_bpe_train],
                                                           [char_valid, small_bpe_valid, big_bpe_valid],
                                                           ["Character-level tokenization", "Small BPE tokenization (10,000)", "Big BPE tokenization (20,000)"]):
        # Reset losses with every new model.
        train_losses, valid_losses = [], []
        
        # Train the model and get the training and validation losses for each epoch.
        model, train_losses, valid_losses = train_model(tokenizer = tokenizer,
                                                        train_loader = train_loader,
                                                        valid_loader = valid_loader,
                                                        device = device,
                                                        name = name,
                                                        C = C,
                                                        n_layers = n_layers,
                                                        n_heads = n_heads,
                                                        ff_dim = ff_dim,
                                                        dropout_rate = dropout_rate,
                                                        T = T,
                                                        epochs = epochs,
                                                        lr = lr)

        # Plot the training and validation losses for this particular model.
        plot_losses(train_losses = train_losses,
                    valid_losses = valid_losses,
                    name = name,
                    save_path = f"plots/losses_{n}.png")
        
        # Save the training and validation losses for this model.
        losses[name] = train_losses, valid_losses

        # Save the model to disk.
        Path("models").mkdir(parents = True, exist_ok = True)
        torch.save(model.state_dict(), f"models/model_{n}.pth")
        print(f"Model #{n} saved to models/model_{n}.pth")

        n += 1
    
    # Plot the training and validation losses for all models for comparison.
    plot_loss_comparison(losses, "plots/loss_comparison.png")
    
        

        

    




    
    
   
    


    
