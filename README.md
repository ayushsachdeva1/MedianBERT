# Median is all you need

## Problem:

The majority of transformer-based pre-trained language models are vocabulary dependent, where each token is mapped to its corresponding vector using an embedding matrix of size V x d (where V is the number of tokens or the vocabulary size and d is the size of each embedding vector). This approach has a few challenges: lack of support for out of vocabulary words and a linear increase in memory usage with increase in vocabulary size. These challenges are a result of the embedding matrix setup. If we could separate the embedding matrix from the vocabulary size, then we could get rid of both of these issues. 

We propose a count-median sketch approach to a token to embedding vector map. We use a trainable k x R x d embedding matrix (where k is the number of independent hash functions we want to use in the count-median sketch, R is the array size we are hashing into, and d is the size of each embedding matrix). When given a token, we hash the token k times into the R-sized array using the hash functions, and compute the element-wise median of the k chosen embedding vectors to be the final embedding vector for the token. The most important aspect of this setup is the independence of the embedding matrix from the vocabulary size. 

## Literature Survey:

Previous probabilistic approaches to tokenizers primarily include using bloom filter based algorithms. Our work has been inspired by Svenstrup, Dan, et al., 2017 [1] who initially proposed the idea of using multiple hash functions to choose from a bucket of candidate embedding vectors and take a weighted average as the final embedding matrix. This work was adapted to be used in the spacy tokenizer [3]. 

Another important work in this domain was Xue, Huiyin, et al., 2022 [2] which adapted the Hash Embedding to a generalized “combination of candidate embedding vectors”. The paper experimented with various ways of combining these vectors to form the final embedding vector such as adding, pooling, and projecting. The paper also used locality sensitive hashing instead of a classical hash functions which was an interesting result. 


## Hypothesis: 

We hypothesize that hash embeddings can be used in BERT, GPT, and other large transformer models, achieving superior performance on NLP benchmarks given an equal amount of training time; furthermore, using the median instead of the mean to aggregate embedding vectors will improve performance.

## Target experimental settings

Common settings and hyperparameters (applies to all models):
Model dimension: 512
Number of encoder layers: 4
Number of attention heads: 4
Max sequence length: 512
Runs per configuration: 3
Batch size: 128 (pre-training), 64 (fine-tuning)
Pre-training objective: masked language modeling (we will introduce a new method to do this that is compatible with hash embeddings, where K hash functions results in K loss function computations)
Experiments for BERT:
Baseline: Standard embeddings (control)
Hash embeddings w/ mean aggregation: 5 hash functions each w/ range 500
Hash embeddings w/ median aggregation: 5 hash functions each w/ range 500
Datasets: Wikipedia and BookCorpus for pretraining, GLUE for evaluation
Evaluation metrics: GLUE score (average accuracy/F1 on GLUE datasets)

[1] Svenstrup, Dan, et al. “Hash Embeddings for Efficient Word Representations.” ArXiv.org, 12 Sep. 2017, https://arxiv.org/abs/1709.03933
[2] Xue, Huiyin, et al. “HashFormers: Towards Vocabulary-Independent Pre-Trained Transformers.” ArXiv.org, 29 Oct. 2022, https://arxiv.org/abs/2210.07904
[3] Miranda, Lester James, et al. “Multi hash embeddings in spaCy.” ArXiv.org, 19 Dec. 2022, https://arxiv.org/abs/2212.09255

Note: Some of the code has been sourced from https://github.com/codertimo/BERT-pytorch