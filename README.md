# General Encoder Retrieval Evaluation


### Environment

Install pytorch accordingly, then `pip install requirements.txt`

Usage: `python run.py --help`


### Evaluate Retrieval

`python run.py --dataset nfcorpus --topk 10 --model BAAI/bge-base-en-v1.5`

- `dataset` is the dataset under [dataset](dataset)
- Results and embeddings will be saved at a newly created directory `evaluation`.


### Evaluate Pairs

`python run_pairs.py --dataset XXX --threshold 0.9 --model XXX`

