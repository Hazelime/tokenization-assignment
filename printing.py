from tokenizers import Tokenizer
from build_tokenizers import CharTokenizer

def print_stats(stats: dict[dict[dict]],
                stat_names: list[str],
                tokenizers: list[Tokenizer or CharTokenizer]) -> None:
    """
    Prints pretty tables for all stat x tokenizer x language combos, like this:
    -------------avg_tokens_per_sent--------------
                |    en    |    tr    |    zh    |
    Char-level  |    81.117|     98.63|    38.546|
    Small BPE   |    68.086|    86.602|     37.98|
    Big BPE     |    35.734|     44.89|    35.103|

    Parameters:
        stats: dict(dict(dict)
            The collection of statistics, ordered as [tokenizer][language code][statistic].
        stat_names: list[str]
            The list of statistic names to print.
        tokenizers: list
            List of tokenizers for which to print stats.
    """
    names = ["Char-level", "Small BPE", "Big BPE"]
    for stat in stat_names:
        # ^ creates centering; - is the fill character; 52 is the total width.
        print(f"{stat:-^52}")
        print(f"{'':12}|{'en':^12}|{'tr':^12}|{'zh':^12}|")
        for name, tokenizer in zip(names, tokenizers):
            # :12.6g means that each cell will be 12 character wide and contain numbers with
            # up to 6 significant figures.
            print(
                f"{name:12}|"
                f"{stats[tokenizer]["en"][stat]:12.6g}|"
                f"{stats[tokenizer]["tr"][stat]:12.6g}|"
                f"{stats[tokenizer]["zh"][stat]:12.6g}|"
            )
        print() # New line after each table.


def print_sample_tokenizations(tokenizers: list, sample_sents: list[str]) -> None:
    """
    Prints pretty tokenization comparisons for given tokenizers and sentences, like this:
    Sample sent #9: 共有14条线路，仅学生上下课期间运营。
    Char-level:     共 有 1 4 条 线 路 ， 仅 学 生 上 下 课 期 间 运 营 。
    Small BPE:      共 有 1 4 条 线 路 ， 仅 学 生 上 下 课 期 间 运 营 。
    Big BPE:        共 有 14 条 线 路 ， 仅 学 生 上 下 课 期 间 运 营 。

    Parameters:
        tokenizers: list
            List of tokenizers to test.
        sample_sents: list[str]
            The samples sentences to tokenize.
    """
    sent_number = 1
    for sent in sample_sents:
        print(f"Sample sent #{sent_number}: {sent}")
        print(f"Char-level: \t{" ".join(tokenizers[0].encode(sent).tokens)}")
        print(f"Small BPE: \t{" ".join(tokenizers[1].encode(sent).tokens)}")
        print(f"Big BPE: \t{" ".join(tokenizers[2].encode(sent).tokens)}\n")
        sent_number += 1