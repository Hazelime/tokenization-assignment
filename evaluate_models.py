from tokenizers import Tokenizer
from build_tokenizers import CharTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import DecoderTransformer
from train_models import get_encoded_dataloader
from printing import print_stats
from hyperparameters import *

def get_npc_on_test(model: nn.Module,
                    test_data: DataLoader,
                    char_count: int,
                    device: torch.device) -> float:
    """
    Calculates the nats per Unicode character (NPC) for a given model on a given test
    dataset. I use NPC instead of BPC since cross-entropy already returns the loss in
    nats, and BPC is just NPC/ln(2) anyway.

    Parameters:
        model: nn.Module
            The trained neural network model to evaluate.
        test_data: DataLoader
            A data loader for the test dataset.
        char_count: int
            The total number of characters in the test dataset.
        device: torch.device
            The device (CPU or GPU) on which to perform the evaluation.
    
    Returns:
        float
            The calculated nats per character (NPC) for the model on the test dataset.
    """
    # Set the model to evaluation mode.
    model.eval()
    total_loss = 0.0

    # Disable gradient computation for evaluation. Apparently, inference_mode() is more
    # efficient than no_grad().
    with torch.inference_mode():
        for x, y in test_data:
            # Move the current batch to the specified device.
            x = x.to(device)
            y = y.to(device)

            # Forward pass through the model to get predictions.
            logits = model(x)

            # Flatten the logits and corresponding targets since CrossEntropyLoss expects
            # the class dimension (V) second. We're making B * T predictions either way.
            # Use nn.functional.cross_entropy to avoid creating a new loss object for each
            # batch. Use reduction = sum to sum the loss over the batch and sequence length.
            # Note that this loss is for both the text AND the sentence boundaries ([EOS]).
            # Since all tokenizers use the same [EOS] token, it's still a fair comparison.
            b, t, v = logits.shape
            loss = nn.functional.cross_entropy(logits.reshape(b * t, v),
                                               y.reshape(b * t),
                                               reduction = "sum")

            total_loss += loss.item() 

    npc = total_loss / char_count if char_count > 0 else float('inf')
    return npc


def char_count(file_path: str) -> int:
    """
    Counts the total number of characters in a given text file.

    Parameters:
        file_path: str
            The path to the text file to be analyzed.
    
    Returns:
        int
            The total number of characters in the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(len(line.strip()) for line in f)


if __name__ == '__main__':
    # Sample syntax:
    # python3 evaluate_models.py 

    test_paths = ["/srv/data/lt2326-h26/a1/test/en.txt", 
                  "/srv/data/lt2326-h26/a1/test/tr.txt", 
                  "/srv/data/lt2326-h26/a1/test/zh.txt"]

    # Pre-calculate the number of characters in each test file for BPC calculation, since
    # they are independent of model and tokenizer. This also avoids counting the underscores
    # from the BPE Metaspace pre-tokenization.
    char_counts = [char_count(test_path) for test_path in test_paths]

    # Load the tokenizers.
    tokenizers = [CharTokenizer().from_file("tokenizers/char_tokenizer.json"), 
                  Tokenizer.from_file("tokenizers/bpe_10000_tokenizer.json"),
                  Tokenizer.from_file("tokenizers/bpe_20000_tokenizer.json")]

    model_paths = ["models/model_1.pth", 
                   "models/model_2.pth", 
                   "models/model_3.pth"]

    # Make sure to use GPUs if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")

    npcs_on_test = {}
    for model_path, tokenizer in zip(model_paths, tokenizers):
        # Create a dictionary entry for each tokenizer unless it already exists.
        if tokenizer not in npcs_on_test: npcs_on_test[tokenizer] = {}

        # Load the model and its weights.
        model = DecoderTransformer(V = len(tokenizer.get_vocab())).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))

        for test_path, char_count in zip(test_paths, char_counts):
            print(f"Evaluating {model_path} on {test_path}...")

            # Encode the current test data with the current tokenizer.
            # T = context length and B = batch size are imported from hyperparameters.py.
            encoded_test = get_encoded_dataloader([test_path], tokenizer, T = T, B = B)

            # Calculate nats per character (NPC) for each model x language test set combo.
            # It's a bit convoluted, but this allows me to reuse printing code.
            lang_code = test_path.split("/")[-1].split(".")[0]  # Extract language code
            npc = get_npc_on_test(model, encoded_test, char_count, device)
            npcs_on_test[tokenizer][lang_code] = {"npc": npc}
        
        # Also calculate NPC across all test sets for the current model x tokenizer combo.
        encoded_test_all = get_encoded_dataloader(test_paths, tokenizer, T = T, B = B)
        total_char_count = sum(char_counts)
        npc_all = get_npc_on_test(model, encoded_test_all, total_char_count, device)
        model_name = model_path.split("/")[-1].split(".")[0]  # Extract model name
        print(f"Overall NPC for {model_name}: {npc_all:.4f}")
        
    # Print the NPCs for each model x language test set combo in a pretty table.
    print_stats(npcs_on_test, ["npc"], tokenizers)

    

