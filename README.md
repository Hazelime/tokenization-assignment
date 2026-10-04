# Assignment 1 - Multilingual Tokenization and Language Modeling
This is a repository used to complete an assignment in the GU course Machine Learning for Statistical NLP: Advanced (LT2326). In a nutshell, the code builds and analyzes one character-level tokenizer and two BPE tokenizers, trains one decoder-only Transformer model per tokenizer, and then evaluates the nats-per-character (NPC) for each model on English, Turkish, and Chinese data.

A pre-experiment checkpoint is available in ```Assignment 1 checkpoint.pdf```, the final report in ```Assignment 1 report.pdf```, and a description of AI use in ```Assignment 1 AI log.pdf```.

Below is how to reproduce the full experiment.

## Build tokenizers
This will build and train three tokenizers, which takes about 1 minute on ```mlt-gpu```. All tokenizers use ```[UNK]``` and ```[EOS]``` special tokens. The BPE tokenizers use Metaspace pre-tokenization.
```
python3 build_tokenizers.py
```

## Analyze tokenizers
This will calculate vocabulary size, average Unicode characters/token, total tokens necessary to encode the training data, and average tokens/sentence for each tokenizer x language combo. It will also provide sample tokenizations of three sentences from each language. Pretty printing is handled by functions in ```printing.py```. (Commented analysis results are recorded in ```tokenizer_analysis.txt```, and pre-experiment expectations are recorded in ```pre-experiment_analysis```.txt.)
```
python3 analyze_tokenizers.py
```

## Train models
This will train one decoder-only Transformer model for each tokenization condition and produce plots of the associated training and validation loss. Hyperparameters are available in ```hyperparameters.py```. Training for 8 epochs takes about 20 minutes on the ```mlt-gpu``` GPUs and results in the models available in the ```models``` folder, with associated training and validation curves ending up in the ```plots``` folder. The model classes themselves are available in ```models.py```, and plotting functions are in ```plotting.py```.
```
python3 train_models.py
```

## Evaluate models
This will output the nats per character (NPC) for each model on the test data, both overall and across different languages.
```
python3 evaluate_models.py
```

## Analyze vocabulary distribution
This will calculate how much of the BPE vocabulary is shared and how much is associated primarily with one of the languages. It will also print various other statistics related to vocabulary distribution.
```
python3 analyze_vocab_distribution.py
```