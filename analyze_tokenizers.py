from tokenizers import Tokenizer
from build_tokenizers import CharTokenizer
from printing import print_stats, print_sample_tokenizations


def get_stats(tokenizer: Tokenizer or CharTokenizer,
              valid_path: str) -> tuple[int, float, int, float]:
    """
    Calculates and returns various tokenization stats based on a tokenizer and a text file.

    Parameters:
        tokenizer
            A tokenizer object (Tokenizer or CharTokenizer) to analyze
        valid_path: str
            A path to the file on which to validate the tokenizer. The file should contain
            one sentence per line.
    
    Returns:
        int
            Vocabulary size
        float
            Average number of Unicode characters per token
        Int
            Total number of tokens needed to encode the validation file
        float
            Average number of tokens per sentence 
    """
    # Get vocabulary size.
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    # Get total number of Unicode characters.
    total_unicode = 0
    for token in vocab:
        total_unicode += len(token)
    
    # Get total number of lines and tokens.
    total_tokens, total_lines = 0, 0
    with open(valid_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            total_tokens += len(tokenizer.encode(line.strip()))
    
    # Return vocabulary size, average number of Unicode characters per token, total number
    # of tokens needed to encode the validation file, and the average number of tokens per
    # sentence. 
    return vocab_size, total_unicode/vocab_size, total_tokens, total_tokens/total_lines


if __name__ == '__main__':
    # Sample syntax:
    # python3 analyze_tokenizers.py 

    valid_paths = ["/srv/data/lt2326-h26/a1/valid/en.txt", 
                   "/srv/data/lt2326-h26/a1/valid/tr.txt", 
                   "/srv/data/lt2326-h26/a1/valid/zh.txt"]

    # Load the tokenizers.
    tokenizers = [CharTokenizer().from_file("tokenizers/char_tokenizer.json"), 
                  Tokenizer.from_file("tokenizers/bpe_10000_tokenizer.json"),
                  Tokenizer.from_file("tokenizers/bpe_20000_tokenizer.json")]

    # Three sample sentences for each language.
    sample_sents = ["In 1536, the Act of Union was passed under Henry's rule which had a long-lasting effect on Wales as a nation.",
                    "Amans and Azoline moved back to France in 1856 where he died in 1888, château de Lévis Saint Nom (78), never having returned to Louisiana.",
                    "The corals will become bleached (lose their colours) and many species that live on and around the reef will be in danger.",
                    "Arşivde yer alan farklı iki parça da 2017'de \"3 Gerald Remix / 24 TSIM 2\" ismiyle Michigan'da bir plak mağazasında satılmıştır.",
                    "Kayseri iline 54 km, Bünyan ilçesine 19 km uzaklıktadır.",
                    "Germantown akademisine devam ettikten sonra 1869-1870 yılları Fransa ve Almanya'da okullarda geçti.",
                    "藝術家會在作品中控制灰分的百分比，若降低灰的比例，控制的程度更高 Ford & Impey, 46-50 。",
                    "這不是外來的概念，傳統是由瑪泰在死亡前以一種被稱為「馬卡加」(mavaega)的意志形式出現，後來則由公共信託辦公室和法律從業者負責處理遺產管理。",
                    "共有14条线路，仅学生上下课期间运营。"]

    # Get stats for how the different tokenizers tokenized the different validation files.
    stats = {}
    for tokenizer in tokenizers:
        # Create a dictionary entry for each tokenizer unless it already exists.
        if tokenizer not in stats: stats[tokenizer] = {}
        
        # Extract stats for every tokenizer x language combo and save them in stats.
        for path in valid_paths:
            vocab_size, avg_unicode_per_token, total_tokens, avg_tokens_per_sent = get_stats(tokenizer = tokenizer,
                                                                                             valid_path = path)
            # Extract language code
            lang_code = path.split("/")[-1].split(".")[0]  
            stats[tokenizer][lang_code] = {
                "vocab_size": vocab_size,
                "avg_unicode_per_token": avg_unicode_per_token,
                "total_tokens": total_tokens,
                "avg_tokens_per_sent": avg_tokens_per_sent
            }
    
    # Print stats and sample tokenizations for all tokenizers.
    print_stats(stats = stats,
                stat_names = ["vocab_size", "avg_unicode_per_token", "total_tokens", "avg_tokens_per_sent"],
                tokenizers = tokenizers)
    print_sample_tokenizations(tokenizers = tokenizers, sample_sents = sample_sents)
                               