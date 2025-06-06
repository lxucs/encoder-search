# General Encoder Retrieval Evaluation

Lightweight research-use framework for fast adaptation.


### Environment

Install pytorch first, then `pip install requirements.txt`

Usage: `python run.py --help`


### Evaluate Retrieval

##### Examples

Use BM25 (only for Chinese for now):

`python run.py --dataset xxx --topk 10 --mode bm25`

Use CLS pooling (default pooling) for BGE models:

`python run.py --dataset nfcorpus --topk 10 --model BAAI/bge-base-en-v1.5`

Use CLS pooling for GTE models:

`python run.py --dataset nfcorpus --topk 10 --model Alibaba-NLP/gte-multilingual-base`

Use SentenceTransformers for Conan-v1 model:

`python run.py --dataset xxx --topk 10 --model TencentBAC/Conan-embedding-v1 --pooling use_sentence_transformer`

Use mean pooling for e5 models, with according prompt templates:

`python run.py --dataset nfcorpus --topk 10 --model intfloat/multilingual-e5-base --pooling mean --max_len 512 --query_template "query: {text}" --candidate_template "passage: {text}"`

Use last token pooling for GTE-Qwen models, with according prompt templates:
    
`python run.py --dataset nfcorpus --topk 10 --model Alibaba-NLP/gte-Qwen2-7B-instruct --pooling last --query_template "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}"`

Notes:
- `dataset` is the dataset under [dataset](dataset)
- Results and embeddings will be saved at a newly created directory `evaluation`.


### Evaluate Pairs

`python run_pairs.py --dataset XXX --threshold 0.9 --model XXX`


### On-the-fly Evaluation

Compute the distance on given queries and candidates.

Usage: `python test.py --help`

Example: `python test.py --data test.json --model BAAI/bge-base-en-v1.5`
