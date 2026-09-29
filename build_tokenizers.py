from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json
from pathlib import Path

class CharEncoding():
    """
    A class to keep encoded characters and their respective tokens together, similar to 
    HuggingFace's tokenizers library.
    """
    def __init__(self, vocab: dict, text: str, unk_token: str = "[UNK]"):
        self.encoding, self.tokens = [], []
        unk_id = vocab[unk_token]
        # Build lists of encoding numbers and actual tokens.
        for char in text:
            token = char if char in vocab else unk_token
            self.tokens.append(token)
            self.encoding.append(vocab.get(char, unk_id))

    def __len__(self):
    """
    Helper function to provide the expected length of a CharEncoding. 
    """
        return len(self.encoding)


class CharTokenizer():
    """
    A simple character-level tokenizer built to mimic how to train and use tokenizers in
    HuggingFace's tokenizers library. Uses an [UNK] token for unseen characters by default.
    """
    def __init__(self, unk_token: str = "[UNK]"):
        self.vocab = dict()
        self.unk_token = unk_token
    
    def train(self, files: list[str]):
        """
        Trains the chracter-level tokenizer on a given list of files. Each token is associated
        with a unique integer index, with the unknown token ([UNK] by default) being 0.

        Parameters:
            files: list[str]
                A list of file paths to train the tokenizer on. Each file should
                contain one sentence per line
        """
        self.vocab[self.unk_token] = 0
        idx = 1  # Start indexing from 1 to reserve 0 for the unknown token.
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    for char in line.strip():
                        if char not in self.vocab:
                            self.vocab[char] = idx
                            idx += 1

    def save(self, path: str):
        """
        Saves the tokenizer (unknown token + vocabulary) to a JSON file for later use.

        Parameters:
            path: str
                The file path where the tokenizer should be saved. The parent directory will be
                created if it doesn't exist.
        """
        # Create the parent directory if it doesn't exist (in case the class is used elsewhere).
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # ensure_ascii = False is necessary to preserve Chinese (and maybe Turkish?)
        # characters in the vocabulary.
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"unk_token": self.unk_token,
                        "vocab": self.vocab},
                        f,
                        indent = 4,
                        ensure_ascii = False) 

    def encode(self, text: str):
        """
        Tokenizes a given string into characters. Unseen characters are replaced with the
        unknown token ([UNK] by default).

        Parameters:
            text: str
                The string to tokenize.
        Returns:
            list[str]
                A list of characters (tokens) from the input string, with unseen characters
                replaced by the unknown token.
        """
        return CharEncoding(self.vocab, text, self.unk_token)
        #return [self.vocab[char] if char in self.vocab else self.vocab[self.unk_token] for char in text]

    def from_file(self, path: str):
        """
        Loads a character-level tokenizer (unknown token + vocabulary) from a JSON file.

        Parameters:
            path: str
                The file path from which to load the tokenizer. The file should be in the
                same format as produced by the save() method.
        Returns:
            CharTokenizer
                The loaded tokenizer.
        """    
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.unk_token = data["unk_token"]  
            self.vocab = data["vocab"]
        return self

    def get_vocab(self):
        """
        Returns the vocabulary.
        """
        return self.vocab


def build_char_tokenizer(train_files: list[str], save_path: str):
    """
    Trains a character-level tokenizer on a given list of files and saves it to a specified path.
    The file will look like this:
    {
        "unk_token": "[UNK]",
        "vocab": {
            "[UNK]": 0,
            "ø": 1,
            "顏": 2,
            ...
            }
    }

    Parameters:
        train_files: list[str]
            A list of file paths to train the tokenizer on. Each file should contain one
            sentence per line.
        save_path: str
            The file path where the trained tokenizer should be saved. 
    """
    char_tokenizer = CharTokenizer(unk_token = "[UNK]")
    char_tokenizer.train(train_files)
    char_tokenizer.save(save_path)


def build_bpe_tokenizer(train_files: list[str], vocab_size: int, save_path: str):
    """
    Trains a BPE tokenizer on a given list of files and saves it to a specified path. Useful
    reference: https://huggingface.co/docs/tokenizers/en/quicktour

    Parameters:
        train_files: list[str]
            A list of file paths to train the tokenizer on. Each file should contain one
            sentence per line.
        vocab_size: int
            The desired vocabulary size for the BPE tokenizer.
        save_path: str
            The file path where the trained tokenizer should be saved.
    """
    # Initialize the BPE tokenizer with an unknown token.
    bpe_tokenizer = Tokenizer(BPE(unk_token = "[UNK]"))
    
    # Set the pre-tokenizer to whitespace so the tokenizer learns subword units within words.
    bpe_tokenizer.pre_tokenizer = Whitespace()
    
    # Create a BPE trainer with special tokens and the specified vocabulary size. (I'm not
    # sure if [CLS], [SEP], [PAD], and [MASK] are necessary for this assignment, but I'm
    # keeping them there just in case.)
    bpe_trainer = BpeTrainer(special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                             vocab_size = vocab_size)
    
    # Train the BPE tokenizer on the training data.
    bpe_tokenizer.train(train_files, bpe_trainer)
    
    # Save the trained BPE tokenizer to the specified path.
    bpe_tokenizer.save(save_path)


if __name__ == '__main__':
    # Sample syntax:
    # python3 build_tokenizers.py 

    # Files are one sentence per line, like this:
    #   Inuit means more than one, one person is an "Inuk".
    #   Sunday edition until his death in 2017.
    #   Chicago: University of Chicago Press, 164–176.
    train_files = ["/srv/data/lt2326-h26/a1/train/en.txt", 
                   "/srv/data/lt2326-h26/a1/train/tr.txt", 
                   "/srv/data/lt2326-h26/a1/train/zh.txt"]
    
    # Create a folder for the tokenizers if it doesn't already exist. exist_ok = True means
    # that no error will be raised if the folder already exists.
    Path("tokenizers").mkdir(exist_ok=True)

    # Train and save character-level tokenizer.
    print("Training and saving character-level tokenizer...")
    build_char_tokenizer(train_files, save_path = "tokenizers/char_tokenizer.json")
    print("Character-level tokenizer saved in tokenizers/char_tokenizer.json\n")

    # Train and save small and big BPE tokenizers.
    print("Training and saving BPE tokenizers...")
    build_bpe_tokenizer(train_files, vocab_size = 2000, save_path = "tokenizers/bpe_2000_tokenizer.json")
    build_bpe_tokenizer(train_files, vocab_size = 10000, save_path = "tokenizers/bpe_10000_tokenizer.json")
    print("BPE tokenizers saved in tokenizers/bpe_2000_tokenizer.json and tokenizers/bpe_10000_tokenizer.json")

