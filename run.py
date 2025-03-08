from typing import Optional
import io_util
import faiss
import os
import numpy as np
from os.path import exists, join
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from argparse import ArgumentParser
from tqdm import tqdm, trange
from model_colbert import ColbertModel
import torch
import logging
import jieba
from transformers import BertModel, AutoTokenizer, AutoModel, AutoModelForSequenceClassification


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()

cn_stopwords = set(io_util.read('resources/cn_stopwords.txt'))


@dataclass
class Searcher:

    save_dir: str
    dataset_name: str

    model_name: str
    pooling_type: str
    normalize: bool
    query_prefix: str
    cand_prefix: str

    do_rerank: bool = False
    reranker_name: str = None

    do_lower_case: bool = True

    def __post_init__(self):
        if self.dataset_name:
            self.model_alias = self.model_name.split('/')[-1]
            self.cand_path = join('dataset', self.dataset_name, f'candidates.jsonl')
            self.query_path = join('dataset', self.dataset_name, f'queries.jsonl')
            assert exists(self.cand_path), 'Dataset does not exist'

            self.cand_emb_path = join(self.save_dir, f'cache.cand.emb.{self.model_alias}.bin')
            self.cand_idx_path = join(self.save_dir, f'idx.{self.dataset_name}.{self.model_alias}.bin')
            self.query_emb_path = join(self.save_dir, f'cache.query.emb.{self.model_alias}.bin')

            self.bm25_idx_path = join(self.save_dir, f'bm25.{self.dataset_name}.{self.model_alias}.bin')
            os.makedirs(self.save_dir, exist_ok=True)

            self.reranker_alias = self.reranker_name.split('/')[-1] if self.reranker_name else None

    def normalize_text(self, text):
        if self.do_lower_case:
            text = text.lower()
        return ' '.join(text.split())

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    @cached_property
    def model(self):
        return AutoModel.from_pretrained(self.model_name).to(self.device)

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    @cached_property
    def reranker(self):
        assert self.reranker_name
        return AutoModelForSequenceClassification.from_pretrained(self.reranker_name).to(self.device)

    @cached_property
    def reranker_tokenizer(self):
        assert self.reranker_name
        return AutoTokenizer.from_pretrained(self.reranker_name)

    @cached_property
    def candidates(self):
        return io_util.read(self.cand_path)

    @cached_property
    def queries(self):
        return io_util.read(self.query_path) if exists(self.query_path) else None

    @classmethod
    def encode(cls, model, tokenizer, lines, pooling_type, normalize, batch_size=16):
        """ Return numpy array. """
        assert pooling_type in ('cls', 'mean')
        single_input = isinstance(lines, str)
        lines = [lines] if single_input else lines

        model.eval()
        length_sorted_idx = np.argsort([-len(line) for line in lines])
        lines_sorted = [lines[idx] for idx in length_sorted_idx]

        all_hidden = []
        for l_i in trange(0, len(lines), batch_size, desc='Encode'):
            batch = lines_sorted[l_i: l_i + batch_size]
            batch = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors='pt').to(model.device)
            with torch.no_grad():
                hidden = model(**batch)['last_hidden_state']  # [bsz, seq_len, hidden]

                if pooling_type == 'cls':
                    hidden = hidden[:, 0]
                else:
                    hidden[~batch['attention_mask'].bool()] = 0
                    hidden = hidden.sum(dim=1) / batch['attention_mask'].sum(dim=1, keepdim=True)

                # Normalize in the end
                if normalize:
                    hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
                all_hidden.append(hidden.float().numpy(force=True))

        # Revert to original order
        all_hidden = np.concatenate(all_hidden, axis=0)  # [num_lines, hidden]
        all_hidden = np.array([all_hidden[idx] for idx in np.argsort(length_sorted_idx)])
        all_hidden = all_hidden[0] if single_input else all_hidden
        return all_hidden

    @classmethod
    def encode_pairs(cls, reranker, tokenizer, pairs, batch_size=32):
        reranker.eval()
        all_probs = []
        for l_i in trange(0, len(pairs), batch_size, desc='Encode', disable=True):
            batch = pairs[l_i: l_i + batch_size]
            batch = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors='pt').to(reranker.device)
            with torch.no_grad():
                logits = reranker(**batch)['logits']
                probs = torch.nn.functional.sigmoid(logits).view(-1)
                all_probs += probs.tolist()
        return all_probs

    @cached_property
    def candidate_has_multi_feature(self):
        return not isinstance(self.candidates[0]['text'], str)

    @cached_property
    def feats(self):
        assert self.candidate_has_multi_feature
        feat2cis = defaultdict(list)
        for c_i, inst in enumerate(self.candidates):
            for feat in inst['text']:
                feat2cis[feat].append(c_i)
        all_feats = list(feat2cis.keys())
        feati2cis = {feat_i: feat2cis[feat] for feat_i, feat in enumerate(all_feats)}
        return all_feats, feati2cis

    @cached_property
    def cand2emb(self):
        text2emb = io_util.read(self.cand_emb_path) if exists(self.cand_emb_path) else {}

        if self.candidate_has_multi_feature:
            all_text, _ = self.feats
        else:
            all_text = [inst['text'] for inst in self.candidates]
        all_text = [self.normalize_text(text) for text in all_text]

        to_embed = list({(self.cand_prefix + text) for text in all_text if (self.cand_prefix + text) not in text2emb})
        if to_embed:
            encoded = self.encode(self.model, self.tokenizer, to_embed, self.pooling_type, self.normalize)
            new_text2emb = {text: emb for text, emb in zip(to_embed, encoded)}
            text2emb |= new_text2emb
            io_util.write(self.cand_emb_path, text2emb)
            print(f'Saved {len(new_text2emb)} new candidate emb to {self.cand_emb_path}')

        cand_emb = np.stack([text2emb[self.cand_prefix + text] for text in all_text], axis=0)
        return cand_emb

    @cached_property
    def query2emb(self):
        text2emb = io_util.read(self.query_emb_path) if exists(self.query_emb_path) else {}
        all_text = [self.normalize_text(inst['query']) for inst in self.queries]

        to_embed = list({(self.query_prefix + text) for text in all_text if (self.query_prefix + text) not in text2emb})
        if to_embed:
            encoded = self.encode(self.model, self.tokenizer, to_embed, self.pooling_type, self.normalize)
            new_text2emb = {text: emb for text, emb in zip(to_embed, encoded)}
            text2emb |= new_text2emb
            io_util.write(self.query_emb_path, text2emb)
            print(f'Saved {len(new_text2emb)} new query emb to {self.query_emb_path}')
        return text2emb

    @cached_property
    def index(self):
        overwrite = True
        if overwrite or not exists(self.cand_idx_path):
            emb = self.cand2emb
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            index = faiss.IndexFlatL2(emb.shape[-1])
            index.add(emb)
            # faiss.write_index(index, self.cand_idx_path)
        else:
            index = faiss.read_index(self.cand_idx_path)
        return index

    @cached_property
    def bm25_index(self):
        from rank_bm25 import BM25Okapi
        overwrite = True
        if overwrite or not exists(self.bm25_idx_path):
            corpus = [jieba.lcut_for_search(inst['text'] if not self.candidate_has_multi_feature else ' '.join(inst['text']))
                      for inst in self.candidates]
            index = BM25Okapi(corpus)
            io_util.write(self.bm25_idx_path, index)
        else:
            index = io_util.read(self.bm25_idx_path)
        return index

    def dense_search(self, query, threshold=None, topk=None):
        """ Sorted by distance. """
        assert query, 'Empty search'
        assert threshold is not None or topk is not None, 'Dense search needs threshold or topk'
        assert threshold is None or topk is None, 'Dense search takes either threshold or topk (not both)'

        query = self.query_prefix + self.normalize_text(query)
        if query in self.query2emb:
            query_emb = self.query2emb[query]
        else:
            query_emb = self.encode(self.model, self.tokenizer, query, self.pooling_type, self.normalize)

        if topk is not None:
            distances, indices = self.index.search(np.expand_dims(query_emb, axis=0), k=topk)
            distances, indices = distances[0], indices[0]
        else:
            limits, distances, indices = self.index.range_search(np.expand_dims(query_emb, axis=0), threshold)
        distances, indices = distances.tolist(), indices.tolist()

        # Adapt multi-feat
        if self.candidate_has_multi_feature:
            _, feati2cis = self.feats
            ci2dist = {}
            for dist, feat_i in zip(distances, indices):
                cis = feati2cis[feat_i]
                for c_i in cis:
                    if c_i not in ci2dist or dist < ci2dist[c_i]:
                        ci2dist[c_i] = dist

            distances, indices = [], []
            for c_i, dist in ci2dist.items():
                distances.append(dist)
                indices.append(c_i)

        # Get results
        results = [self.candidates[c_i] | {'idx': c_i, 'distance': dist}
                   for dist, c_i in zip(distances, indices)]

        # Rule
        for r in results:
            r['distance'] = r['distance'] if r['text'] else float('inf')

        # Sort
        results = sorted(results, key=lambda v: v['distance'])
        for i, inst in enumerate(results):
            inst['rank'] = i
        return results

    def bm25_search(self, text):
        """ Sorted by distance. """
        text = self.normalize_text(text)
        assert text, 'Empty search'

        indices = None
        for term in jieba.lcut(text):
            if term not in cn_stopwords:
                scores = self.bm25_index.get_scores([term]).tolist()
                term_indices = {idx for idx, score in enumerate(scores) if score > 1e-3}
                if indices is None:
                    indices = term_indices
                else:
                    indices &= term_indices  # Take overlap

        results = [self.candidates[idx] | {'idx': idx, 'distance': 0.01}
                   for idx in indices]

        # Rule
        for r in results:
            r['distance'] = r['distance'] if r['text'] else float('inf')

        # Sort
        results = sorted(results, key=lambda v: v['distance'])
        for i, inst in enumerate(results):
            inst['rank'] = i
        return results

    def rerank(self, query, results, threshold, rerank_only_above):
        if not results:
            return results
        query = self.normalize_text(query)  # For rerank, do not use prefix
        candidates = [self.normalize_text(r['text']) for r in results]

        pairs = [[query, cand] for cand in candidates]
        rerank_scores = self.encode_pairs(self.reranker, self.reranker_tokenizer, pairs)

        if rerank_only_above is not None:
            rerank_scores = [(score if r['distance'] > rerank_only_above else 100) for score, r in zip(rerank_scores, results)]

        new_results = [(r | {'rerank_score': score}) for score, r in zip(rerank_scores, results) if score >= threshold]
        if len(new_results) != len(results):
            print(f'Rerank: filtered out {len(results) - len(new_results)} results')
            filtered_out_results = [(r | {'rerank_score': score}) for score, r in zip(rerank_scores, results) if score < threshold]
            print(f'Query: {query}')
            for r in filtered_out_results:
                print(r)
        results = new_results
        return results


@dataclass
class ColbertSearcher(Searcher):

    use_simple_query: bool = False
    use_colbert_linear: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.dataset_name:
            self.cand_emb_path = join(self.save_dir, f'colbert.cache.cand.emb.{self.model_alias}.bin')
            self.cand_idx_path = join(self.save_dir, f'colbert.idx.{self.dataset_name}.{self.model_alias}.bin')
            self.query_emb_path = join(self.save_dir, f'colbert.cache.query.emb.{self.model_alias}.bin')

    @cached_property
    def model(self):
        return ColbertModel(self.model_name, use_linear=self.use_colbert_linear)

    @classmethod
    def encode(cls, model, tokenizer, lines, use_pooled_hidden=False, batch_size=16):
        """ Return list of numpy array. """
        single_input = isinstance(lines, str)
        lines = [lines] if single_input else lines

        model.eval()
        all_hidden = []
        for l_i in trange(0, len(lines), batch_size, desc='Encode'):
            batch_lines = lines[l_i: l_i + batch_size]
            batch = tokenizer(batch_lines, truncation=True, padding=True, max_length=512, return_tensors='pt').to(model.device)
            with torch.no_grad():
                pooled_hidden, colbert_hidden = model.encode(batch, remove_colbert_padding=True)

                if use_pooled_hidden:
                    hidden = pooled_hidden.float().numpy(force=True)
                    all_hidden.append(hidden)
                else:
                    for line, line_hidden in zip(batch_lines, colbert_hidden):
                        line_hidden = line_hidden.float().numpy(force=True)
                        all_hidden.append(line_hidden)

        if use_pooled_hidden:
            all_hidden = np.concatenate(all_hidden, axis=0)  # [num_lines, hidden]
        all_hidden = all_hidden[0] if single_input else all_hidden
        return all_hidden  # [seq_len_wo_pad, hidden] for each line; or [num_lines, hidden]

    @cached_property
    def cand2emb(self):
        text2emb = io_util.read(self.cand_emb_path) if exists(self.cand_emb_path) else {}

        to_embed = [(self.cand_prefix + inst['text']) for inst in self.candidates if (self.cand_prefix + inst['text']) not in text2emb]
        if to_embed:
            encoded = self.encode(self.model, self.tokenizer, to_embed, use_pooled_hidden=False)
            new_text2emb = {text: emb for text, emb in zip(to_embed, encoded)}
            text2emb |= new_text2emb
            io_util.write(self.cand_emb_path, text2emb)
            print(f'Saved {len(new_text2emb)} new candidate emb to {self.cand_emb_path}')

        cand_emb = np.concatenate([text2emb[self.cand_prefix + inst['text']] for inst in self.candidates], axis=0)
        toki2ci = [c_i for c_i, inst in enumerate(self.candidates) for _ in range(text2emb[self.cand_prefix + inst['text']].shape[0])]
        return cand_emb, toki2ci

    @cached_property
    def query2emb(self):
        text2emb = io_util.read(self.query_emb_path) if exists(self.query_emb_path) else {}
        queries_to_embed = [(self.query_prefix + inst['query']) for inst in self.queries if (self.query_prefix + inst['query']) not in text2emb]
        if queries_to_embed:
            encoded = self.encode(self.model, self.tokenizer, queries_to_embed, use_pooled_hidden=self.use_simple_query)
            new_text2emb = {text: emb for text, emb in zip(queries_to_embed, encoded)}
            text2emb |= new_text2emb
            io_util.write(self.query_emb_path, text2emb)
            print(f'Saved {len(new_text2emb)} new query emb to {self.query_emb_path}')
        return text2emb

    def dense_search(self, query, threshold=None, topk=None, query_pooling='mean'):
        """ Sorted by distance. """
        assert query_pooling in ('mean', 'min')
        assert query, 'Empty search'
        assert threshold is not None or topk is not None, 'Dense search needs threshold or topk'
        assert threshold is None or topk is None, 'Dense search takes either threshold or topk (not both)'
        max_threshold = 2  # For colbert, token-level search is full search

        query = self.query_prefix + self.normalize_text(query)
        if query in self.query2emb:
            query_emb = self.query2emb[query]
        else:
            query_emb = self.encode(self.model, self.tokenizer, query, use_pooled_hidden=self.use_simple_query)
        # Handle both simple and colbert query
        if len(query_emb.shape) == 1:
            query_emb = np.expand_dims(query_emb, axis=0)

        limits, distances, indices = self.index.range_search(query_emb, max_threshold)
        cand_emb, candtoki2ci = self.cand2emb
        num_q_toks = query_emb.shape[0]

        qtoki2ci2dist = []
        for q_tok_i in range(num_q_toks):
            start, end = limits[q_tok_i], limits[q_tok_i + 1]
            qtok_distances, qtok_indices = distances[start: end].tolist(), indices[start: end].tolist()
            ci2dist = {}
            for dist, cand_tok_i in zip(qtok_distances, qtok_indices):
                c_i = candtoki2ci[cand_tok_i]
                if c_i not in ci2dist or dist < ci2dist[c_i]:
                    ci2dist[c_i] = dist
            qtoki2ci2dist.append(ci2dist)

        ci2qtoki2dist = defaultdict(dict)
        for q_tok_i, ci2dist in enumerate(qtoki2ci2dist):
            for c_i, dist in ci2dist.items():
                ci2qtoki2dist[c_i][q_tok_i] = dist

        if query_pooling == 'min':
            ci2finaldist = {c_i: min(qtoki2dist.values()) for c_i, qtoki2dist in ci2qtoki2dist.items()}
        elif query_pooling == 'mean':
            ci2finaldist = {c_i: ((sum(qtoki2dist.values()) + max_threshold * (num_q_toks - len(qtoki2dist))) / num_q_toks) for c_i, qtoki2dist in ci2qtoki2dist.items()}
        else:
            raise ValueError(query_pooling)

        distances, indices = [], []
        for c_i, dist in ci2finaldist.items():
            if dist <= threshold:
                distances.append(dist)
                indices.append(c_i)

        # Get results
        results = [self.candidates[c_i] | {'idx': c_i, 'distance': dist} for dist, c_i in zip(distances, indices)]

        # Rule
        for r in results:
            r['distance'] = r['distance'] if r['text'] else float('inf')

        # Sort
        results = sorted(results, key=lambda v: v['distance'])
        for i, inst in enumerate(results):
            inst['rank'] = i

        # Apply threshold and topk after sort
        if topk is not None:
            results = results[:topk]
        else:
            results = [r for r in results if r['distance'] <= threshold]
        return results


@dataclass
class Evaluator:

    save_dir: str
    dataset_path: str

    model_name: str
    pooling_type: str
    normalize: bool
    query_prefix: str
    cand_prefix: str

    is_colbert: bool = False
    use_simple_colbert_query: bool = False
    use_colbert_linear: bool = True

    query_threshold: Optional[float] = None
    topk: Optional[int] = None
    mode: str = 'dense'

    do_rerank: bool = False
    reranker_name: Optional[str] = None
    rerank_threshold: Optional[float] = None
    rerank_only_above: Optional[float] = None

    gold_score: Optional[int] = None

    def __post_init__(self):
        assert self.mode in ('dense', 'bm25')
        if self.mode == 'dense':
            assert self.query_threshold is not None or self.topk is not None, 'Dense search requires threshold or topk'
            assert self.query_threshold is None or self.topk is None, 'Dense search takes threshold or topk (not both)'

        if not self.do_rerank:
            self.reranker_name = self.rerank_threshold = self.rerank_only_above = None

        if self.is_colbert:
            self.searcher = ColbertSearcher(self.save_dir, self.dataset_path, self.model_name, self.pooling_type, self.normalize, self.query_prefix, self.cand_prefix,
                                            use_simple_query=self.use_simple_colbert_query, use_colbert_linear=self.use_colbert_linear)
        else:
            self.searcher = Searcher(self.save_dir, self.dataset_path, self.model_name, self.pooling_type, self.normalize, self.query_prefix, self.cand_prefix,
                                     do_rerank=self.do_rerank, reranker_name=self.reranker_name)

        self.dataset_name = self.searcher.dataset_name
        self.model_alias = self.searcher.model_alias
        self.reranker_alias = self.searcher.reranker_alias
        if self.mode == 'dense':
            th_or_topk = f'th{self.query_threshold}' if self.query_threshold is not None else f'top{self.topk}'
            rerank = f'.rerank{self.rerank_threshold}.{self.reranker_alias}' if self.do_rerank else ''
            self.result_path = join(self.save_dir, f'{"colbert." if self.is_colbert else ""}results.{self.dataset_name}.{self.model_alias}.{th_or_topk}{rerank}.json')
        else:
            self.result_path = join(self.save_dir, f'results.{self.dataset_name}.bm25.json')
        self.report_path = self.result_path.replace('results.', 'report.')

    def get_results(self):
        query_insts = self.searcher.queries
        assert query_insts, f'No queries for dataset {self.dataset_name}'
        if self.mode == 'dense':
            assert self.searcher.query2emb is not None and self.searcher.cand2emb is not None
        else:
            assert self.searcher.bm25_index is not None

        # Search
        for inst in tqdm(query_insts, desc='Search', disable=False):
            inst['mode'] = self.mode
            if self.mode == 'dense':
                inst['topk'] = self.topk
                inst['query_threshold'] = self.query_threshold
                inst['query_results'] = self.searcher.dense_search(inst['query'], threshold=inst['query_threshold'], topk=inst['topk'])
            else:
                inst['topk'] = None
                inst['query_threshold'] = float('inf')
                inst['query_results'] = self.searcher.bm25_search(inst['query'])

        # Rerank
        if self.do_rerank:
            assert self.mode == 'dense' and self.reranker_name
            for inst in tqdm(query_insts, desc='Rerank', disable=True):
                inst['query_results'] = self.searcher.rerank(inst['query'], inst['query_results'], self.rerank_threshold, self.rerank_only_above)
                inst['rerank_threshold'] = self.rerank_threshold
                inst['rerank_only_above'] = self.rerank_only_above

        # Get metrics
        results, ds2metric2score = self.get_metrics(query_insts, gold_score=self.gold_score)

        # Save
        io_util.write(self.result_path, results)
        print(f'Saved {len(results)} query results to {self.result_path}')

        # Save report
        if self.mode == 'dense':
            report = self.get_report(results, self.searcher.candidates)
            io_util.write(self.report_path, report)
            print(f'Saved report to {self.report_path}')
        return results, ds2metric2score

    @classmethod
    def compute_reciprocal_rank(cls, pred_ids, gold_ids):
        """ pred_ids is sorted. """
        if not gold_ids:
            return None
        gold_ids = set(gold_ids)
        rank = None
        for r, id_ in enumerate(pred_ids):
            if id_ in gold_ids:
                rank = r + 1
                break
        return (1 / rank) if rank else 0

    @classmethod
    def compute_average_precision(cls, pred_ids, gold_ids):
        """ pred_ids is sorted. """
        if not gold_ids:
            return None
        gold_ids = set(gold_ids)
        precisions, curr_hit = [], 0
        for i, id_ in enumerate(pred_ids):
            if id_ in gold_ids:
                curr_hit += 1
                precisions.append(curr_hit / (i + 1))
            if curr_hit == len(gold_ids):
                break  # Early stop
        ap = (sum(precisions) / len(precisions)) if precisions else 0
        return ap

    @classmethod
    def compute_ndcg(cls, pred_ids, goldid2score):
        """ pred_ids is sorted. """
        pred_scores = np.array([(goldid2score.get(id_, 0)) for id_ in pred_ids])
        gold_scores = np.array(sorted(goldid2score.values(), reverse=True)[:len(pred_scores)])
        pred_dcg = (pred_scores / np.log2(np.arange(2, len(pred_scores) + 2))).sum().item()
        gold_dcg = (gold_scores / np.log2(np.arange(2, len(gold_scores) + 2))).sum().item()
        return (pred_dcg / gold_dcg) if gold_dcg else None

    @classmethod
    def compute_pair_recall(cls, pred_ids, gold_ids):
        pred_ids, gold_ids = set(pred_ids), set(gold_ids)
        return len(gold_ids & pred_ids), len(gold_ids)

    @classmethod
    def compute_pair_precision(cls, pred_ids, gold_ids):
        pred_ids, gold_ids = set(pred_ids), set(gold_ids)
        return len(gold_ids & pred_ids), len(pred_ids)

    @classmethod
    def compute_query_recall(cls, pred_ids, gold_ids):
        pred_ids, gold_ids = set(pred_ids), set(gold_ids)
        return (len(gold_ids & pred_ids) / len(gold_ids)) if gold_ids else None

    @classmethod
    def compute_query_precision(cls, pred_ids, gold_ids):
        pred_ids, gold_ids = set(pred_ids), set(gold_ids)
        return (len(gold_ids & pred_ids) / len(pred_ids)) if pred_ids else None

    @classmethod
    def compute_f_score(cls, p, r, beta=1.0):
        beta_squared = beta ** 2
        return (1 + beta_squared) * (p * r) / ((beta_squared * p) + r) if p or r else 0

    @classmethod
    def compute_query_hit(cls, pred_ids, gold_ids):
        pred_ids, gold_ids = set(pred_ids), set(gold_ids)
        return float(len(pred_ids & gold_ids) > 0)

    @classmethod
    def finalize_metrics(cls, query_metric2score, times100=False):
        """ Compute average. """
        beta = 0.5
        for metric in query_metric2score.keys():
            scores = query_metric2score[metric]
            query_metric2score[metric] = (sum(scores) / len(scores) * (100 if times100 else 1)) if scores else 0
            print(f'Query evaluation: {metric} = {query_metric2score[metric]:.2f}')

        query_metric2score['query_f1'] = cls.compute_f_score(query_metric2score['query_precision'], query_metric2score['query_recall'])
        query_metric2score[f'query_f{beta}'] = cls.compute_f_score(query_metric2score['query_precision'], query_metric2score['query_recall'], beta=beta)
        print(f'Query evaluation: query_f1 = {query_metric2score["query_f1"]:.2f}')
        print(f'Query evaluation: query_f{beta} = {query_metric2score[f"query_f{beta}"]:.2f}')

        query_metric2score['pair_f1'] = cls.compute_f_score(query_metric2score['pair_precision'], query_metric2score['pair_recall'])
        query_metric2score[f'pair_f{beta}'] = cls.compute_f_score(query_metric2score['pair_precision'], query_metric2score['pair_recall'], beta=beta)
        print(f'Query evaluation: pair_f1 = {query_metric2score["pair_f1"]:.2f}')
        print(f'Query evaluation: pair_f{beta} = {query_metric2score[f"pair_f{beta}"]:.2f}')

        info = "\t".join(f"{score:.2f}" for _, score in query_metric2score.items())
        print(f'Query: {info}')
        return query_metric2score

    @classmethod
    def get_metrics(cls, insts, gold_score=None, query_threshold=None, rerank_threshold=None):
        if gold_score is None:
            print(f'Using all positives as gold')
        else:
            print(f'Using gold_score>={gold_score} positives as gold')
        if query_threshold:
            print(f'Override query_threshold as {query_threshold}')
        if rerank_threshold:
            print(f'Override rerank_threshold as {rerank_threshold}')
        print()

        query_threshold = query_threshold or insts[0]['query_threshold']  # Can be None
        topk = insts[0]['topk']  # Can be None
        assert query_threshold is not None or topk is not None, 'Results should have either threshold or topk'
        metric_suffix = f'@th{query_threshold}' if query_threshold is not None else f'@top{topk}'

        # Get metrics
        for inst in insts:
            gold_ids = [pos['id'] for pos in inst['positives'] if not gold_score or pos['score'] >= gold_score]
            goldid2score = {pos['id']: pos['score'] for pos in inst['positives']}

            inst['query_threshold'] = query_threshold
            if rerank_threshold:
                inst['rerank_threshold'] = rerank_threshold
            query_results = [r for r in inst['query_results'] if (query_threshold is None or r['distance'] <= query_threshold) and (rerank_threshold is None or r['rerank_score'] >= rerank_threshold)]

            result_ids = [r['id'] for r in query_results]
            rr_score = cls.compute_reciprocal_rank(result_ids, gold_ids)
            ap_score = cls.compute_average_precision(result_ids, gold_ids)
            ndcg_score = cls.compute_ndcg(result_ids, goldid2score)
            hit_score = cls.compute_query_hit(result_ids, gold_ids)
            pair_recall = cls.compute_pair_recall(result_ids, gold_ids)
            pair_precision = cls.compute_pair_precision(result_ids, gold_ids)
            query_recall = cls.compute_query_recall(result_ids, gold_ids)
            query_precision = cls.compute_query_precision(result_ids, gold_ids)

            inst['metric_suffix'] = metric_suffix
            inst['query_metrics'] = {f'reciprocal_rank {metric_suffix}': rr_score,
                                     f'average_precision {metric_suffix}': ap_score,
                                     f'ndcg {metric_suffix}': ndcg_score,
                                     f'hit {metric_suffix}': hit_score,
                                     f'query_precision': query_precision,
                                     f'query_recall': query_recall,
                                     f'pair_precision': pair_precision,
                                     f'pair_recall': pair_recall}

            result_ids, gold_ids = set(result_ids), set(gold_ids)
            for target in (inst['positives'] + inst['negatives']):
                target[f'recall'] = target['id'] in result_ids
            for r in inst['query_results']:
                r['is_positive'] = r['id'] in gold_ids

        # Stats per dataset
        ds2metric2score = defaultdict(dict)
        for inst in insts:
            ds = inst['by']
            for metric, score in inst['query_metrics'].items():
                if metric not in ds2metric2score[ds]:
                    ds2metric2score[ds][metric] = []

                if not isinstance(score, (list, tuple)):  # Query-level metric
                    if score is not None:  # Exclude /0 cases
                        ds2metric2score[ds][metric].append(score)
                else:  # Pair-level metric
                    ds2metric2score[ds][metric] += ([1] * score[0] + [0] * (score[1] - score[0]))

        for ds in ds2metric2score.keys():
            print(f'Metrics per dataset {ds}:')
            ds2metric2score[ds] = cls.finalize_metrics(ds2metric2score[ds], times100=True)
            print('=' * 20)

        # Stats overall
        if len(ds2metric2score) > 1:
            print(f'Metrics overall:')
            query_metric2score = {}
            for _, metric2score in ds2metric2score.items():
                for metric, score in metric2score.items():
                    if metric not in query_metric2score:
                        query_metric2score[metric] = []

                    if not isinstance(score, list):  # Query-level metric
                        query_metric2score[metric].append(score)
                    else:  # Pair-level metric
                        query_metric2score[metric] += ([1] * score[0] + [0] * (score[1] - score[0]))
            query_metric2score = cls.finalize_metrics(query_metric2score)
            ds2metric2score['average'] = query_metric2score
        else:
            ds2metric2score['average'] = ds2metric2score[insts[0]['by']]
        return insts, ds2metric2score

    @classmethod
    def get_report(cls, results, candidates=None):
        cid2text = {inst['id']: inst['text'] for inst in candidates}
        report = []
        for inst in results:
            over_recall = [{'id': r['id'], 'text': r['text'], 'distance': r['distance'], 'rerank': r.get('rerank_score', None)}
                           for r in inst['query_results'] if not r['is_positive']]
            need_recall = [{'id': r['id'], 'text': r['text'] if 'text' in r else cid2text[r['id']]}
                           for r in inst['positives'] if not r['recall']]
            p = inst['query_metrics']['pair_precision']
            r = inst['query_metrics']['pair_recall']
            report.append({'id': inst['id'], 'query': inst['query'],
                           'precision': (f'{p[0] / p[1] * 100:.2f}%' if p[1] else None, p),
                           'recall': (f'{r[0] / r[1] * 100:.2f}%' if r[1] else None, r),
                           'over_recall': over_recall, 'need_recall': need_recall})
        return report


def tune_hyperparameters(result_path, gold_score):
    thresholds = [(v / 1000) for v in range(800, 1400, 25)]
    # thresholds = [(v / 1000) for v in range(500, 900, 25)]

    scores = []
    for threshold in thresholds:
        results = io_util.read(result_path)
        _, ds2metric2score = Evaluator.get_metrics(results, gold_score=gold_score, query_threshold=threshold)
        f1 = ds2metric2score['average'][f'query_f1']
        scores.append((f1, (threshold,)))
    scores = sorted(scores)
    for score, (threshold,) in scores:
        print(f'Score: {score:.2f} | threshold: {threshold}')


def main():
    parser = ArgumentParser('Evaluate Retrieval')
    parser.add_argument('--dataset', type=str, help='Dataset under ./dataset', required=True)
    parser.add_argument('--model', type=str, help='Model name or path', default='BAAI/bge-base-en-v1.5')
    parser.add_argument('--pooling', type=str, help='Encoder pooling style', default='cls', choices=['cls', 'mean'])
    parser.add_argument('--normalize', type=int, help='Whether normalize emb', default=1)
    parser.add_argument('--query_prefix', type=str, help='query_prefix', default='')
    parser.add_argument('--cand_prefix', type=str, help='cand_prefix', default='')
    parser.add_argument('--is_colbert', help='Use colbert retrieval', action='store_true')
    parser.add_argument('--use_simple_colbert_query', help='Use simple query emb for colbert', action='store_true')
    parser.add_argument('--disable_colbert_linear', help='Disable linear layer for colbert', action='store_true')
    parser.add_argument('--threshold', type=float, help='Search threshold', default=None)
    parser.add_argument('--topk', type=int, help='Search topk', default=None)
    parser.add_argument('--mode', type=str, help='Search mode', default='dense', choices=['dense', 'exact'])
    parser.add_argument('--do_rerank', help='Do rerank', action='store_true')
    parser.add_argument('--reranker_name', type=str, help='Reranker name or path', default=None)
    parser.add_argument('--rerank_threshold', type=float, help='Rerank threshold', default=None)
    parser.add_argument('--rerank_only_above', type=float, help='Rerank only on above-distance', default=None)
    parser.add_argument('--result_path', type=str, help='Saved result path to compute metrics', default=None)
    args = parser.parse_args()

    if args.result_path:
        results = io_util.read(args.result_path)
        print(f'Evaluation {len(results)} results from {args.result_path}\n')
        Evaluator.get_metrics(results, query_threshold=args.threshold, rerank_threshold=args.rerank_threshold)
    else:
        evaluator = Evaluator('evaluation', args.dataset, args.model, args.pooling, bool(args.normalize), args.query_prefix, args.cand_prefix,
                              is_colbert=args.is_colbert, use_simple_colbert_query=args.use_simple_colbert_query, use_colbert_linear=not args.disable_colbert_linear,
                              query_threshold=args.threshold, topk=args.topk, mode=args.mode,
                              do_rerank=args.do_rerank, reranker_name=args.reranker_name, rerank_threshold=args.rerank_threshold, rerank_only_above=args.rerank_only_above)
        evaluator.get_results()


if __name__ == '__main__':
    main()
