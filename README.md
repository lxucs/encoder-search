# General Encoder Retrieval Evaluation

### Evaluate Retrieval

`python run.py --dataset nfcorpus --model BAAI/bge-base-en-v1.5 --topk 10`

- `dataset` is the dataset under [dataset](dataset)。
- Results and embeddings will be saved at the `evaluation` directory.


### Evaluate Pairs

`python run_pairs.py --dataset XXX --model XXX --threshold 0.9`

