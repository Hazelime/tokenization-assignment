from tokenizers import Tokenizer
from build_tokenizers import CharTokenizer
from collections import defaultdict, Counter
from pathlib import Path


def get_token_counts(tokenizer: Tokenizer or CharTokenizer,
                     train_paths: list[str]) -> defaultdict[Counter]:
    """
    Counts the number of times each token appears in the training data for a given tokenizer.

    Parameters:
        tokenizer
            A tokenizer object (Tokenizer or CharTokenizer) to analyze
        train_paths: list[str]
            A list of file paths to the training data. Each file should contain one sentence
            per line.
    
    Returns:
        defaultdict[Counter]
            A dictionary where each key is a token and each value is a Counter mapping
            language codes to the number of times that token appears in the training data
            for that language.
    """
    token_counts = defaultdict(Counter)
    for train_path in train_paths:
        # Get the language code from the file path.
        #lang_code = train_path.split("/")[-1].split(".")[0]
        lang_code = Path(train_path).stem
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = tokenizer.encode(line.strip()).tokens
                for token in tokens:
                    # Count the number of times each token appears in the training data.
                    token_counts[token][lang_code] += 1
    return token_counts


def get_token_classifications(token_counts: defaultdict[Counter]) -> dict[str, list[str]]:
    """
    Classifies tokens as belonging to a language based on their counts in the training data.
    A token is classified as belonging to a language if at least 30% of the occurrences of
    that token are in that language. This allows for two-way classifications and (very 
    rarely) three-way classifications.

    Parameters:
        token_counts: defaultdict[Counter]
            A dictionary where each key is a token and each value is a Counter mapping
            language codes to the number of times that token appears in the training
            data for that language.
    
    Returns:
        dict[str, [list[str]]]
            A dictionary where each key is a token and each value is a list of languages
            that the token is classified as belonging to.
    """
    token_classifications = {}
    for token, counts in token_counts.items():
        total_count = sum(counts.values())
        classifications = []
        for lang_code, count in counts.items():
            # A token is classified as belonging to a language if at least 30% of the
            # occurrences of that token are in that language. 
            if count / total_count >= 0.3:
                classifications.append(lang_code)
        token_classifications[token] = classifications
    return token_classifications


def get_token_distribution(token_classifications: dict[str, list[str]]) -> list[float]:
    """
    Calculates the language distribution of tokens based on their classifications.

    Parameters:
        token_classifications: dict[str, list[str]]
            A dictionary where each key is a token and each value is a list of languages
            that the token is classified as belonging to.

    Returns:
        list[float]
            A list containing the percentage of tokens that are unique to each language
            and the percentage of tokens that are shared across languages.
    """
    counts = Counter()
    shared_count = 0

    for classifications in token_classifications.values():
        if len(classifications) == 1:
            counts[classifications[0]] += 1
        elif len(classifications) > 1:
            shared_count += 1

    total_tokens = len(token_classifications)

    distribution = [counts["en"] / total_tokens * 100,
                    counts["tr"] / total_tokens * 100,
                    counts["zh"] / total_tokens * 100,
                    shared_count / total_tokens * 100]

    return distribution


if __name__ == '__main__':
    # Sample syntax:
    # python3 analyze_vocab_distribution.py 

    train_paths = ["/srv/data/lt2326-h26/a1/train/en.txt", 
                   "/srv/data/lt2326-h26/a1/train/tr.txt", 
                   "/srv/data/lt2326-h26/a1/train/zh.txt"]

    # Load the BPE tokenizers.
    tokenizers = [Tokenizer.from_file("tokenizers/bpe_10000_tokenizer.json"),
                  Tokenizer.from_file("tokenizers/bpe_20000_tokenizer.json")]
    
    print("Counting small BPE token occurrences in each language...")
    small_bpe_token_counts = get_token_counts(tokenizers[0], train_paths)
    print("Counting big BPE token occurrences in each language...")
    big_bpe_token_counts = get_token_counts(tokenizers[1], train_paths)

    print("Classifying tokens based on their distributions across languages...\n")
    small_bpe_classifications = get_token_classifications(small_bpe_token_counts)
    big_bpe_classifications = get_token_classifications(big_bpe_token_counts)

    # Calculate the distribution of tokens based on their language classifications.
    small_bpe_distribution = get_token_distribution(small_bpe_classifications)
    big_bpe_distribution = get_token_distribution(big_bpe_classifications)
    print(f"Small BPE token distribution: EN: {small_bpe_distribution[0]:.2f}%, TR: {small_bpe_distribution[1]:.2f}%, ZH: {small_bpe_distribution[2]:.2f}%, shared: {small_bpe_distribution[3]:.2f}%")
    print(f"Big BPE token distribution: EN: {big_bpe_distribution[0]:.2f}%, TR: {big_bpe_distribution[1]:.2f}%, ZH: {big_bpe_distribution[2]:.2f}%, shared: {big_bpe_distribution[3]:.2f}%\n")

    # Print tokens that have approximately equal distribution across all three languages and
    # occur more than once in each language.
    for token in small_bpe_classifications:
        if len(small_bpe_classifications[token]) > 2 and small_bpe_token_counts[token]["en"] > 1:
            print(f"Small BPE token '{token}' was assigned to three languages: EN: {small_bpe_token_counts[token].get('en', 0)}, TR: {small_bpe_token_counts[token].get('tr', 0)}, ZH: {small_bpe_token_counts[token].get('zh', 0)}")
    for token in big_bpe_classifications:
        if len(big_bpe_classifications[token]) > 2 and big_bpe_token_counts[token]["en"] > 1:
            print(f"Big BPE token '{token}' was assigned to three languages. EN: {big_bpe_token_counts[token].get('en', 0)}, TR: {big_bpe_token_counts[token].get('tr', 0)}, ZH: {big_bpe_token_counts[token].get('zh', 0)}")
    print()

    # Print tokens that are shared between English and Turkish, but not Chinese.
    for token in small_bpe_classifications:
        if len(small_bpe_classifications[token]) > 1 and \
           small_bpe_token_counts[token]["en"] > 1 and \
           small_bpe_token_counts[token]["tr"] > 1 and \
           small_bpe_token_counts[token]["zh"] < 1:
            print(f"Small BPE token '{token}' was shared between EN and TR: EN: {small_bpe_token_counts[token].get('en', 0)}, TR: {small_bpe_token_counts[token].get('tr', 0)}, ZH: {small_bpe_token_counts[token].get('zh', 0)}")
    for token in big_bpe_classifications:
        if len(big_bpe_classifications[token]) > 1 and \
           big_bpe_token_counts[token]["en"] > 1 and \
           big_bpe_token_counts[token]["tr"] > 1 and \
           big_bpe_token_counts[token]["zh"] < 1:
            print(f"Big BPE token '{token}' was shared between EN and TR: EN: {big_bpe_token_counts[token].get('en', 0)}, TR: {big_bpe_token_counts[token].get('tr', 0)}, ZH: {big_bpe_token_counts[token].get('zh', 0)}")
    print()

    # Print the percentage of tokens shared by English and Turkish only.
    shared_en_tr = 0
    for token in small_bpe_classifications:
        if len(small_bpe_classifications[token]) > 1 and \
           "zh" not in small_bpe_classifications[token]: 
            shared_en_tr += 1 
    print(f"Percentage of Small BPE tokens shared by EN and TR only: {(shared_en_tr / len(small_bpe_classifications) * 100):.2f}%")
    shared_en_tr = 0
    for token in big_bpe_classifications:
        if len(big_bpe_classifications[token]) > 1 and \
           "zh" not in big_bpe_classifications[token]: 
            shared_en_tr += 1 
    print(f"Percentage of Big BPE tokens shared by EN and TR only: {(shared_en_tr / len(big_bpe_classifications) * 100):.2f}%")