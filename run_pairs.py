from typing import Optional
import io_util
import os
import numpy as np
from os.path import exists, join
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from argparse import ArgumentParser
import torch
import logging
import jieba
from run import Searcher as EvalSearcher
from transformers import BertModel, AutoTokenizer, AutoModel, AutoModelForSequenceClassification


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


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
            self.pair_path = join('dataset_pairs', f'{self.dataset_name}.jsonl')
            assert exists(self.pair_path)

            self.cand_emb_path = join(self.save_dir, f'cache.cand.emb.{self.model_alias}.bin')
            self.query_emb_path = join(self.save_dir, f'cache.query.emb.{self.model_alias}.bin')
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
    def insts(self):
        return io_util.read(self.pair_path)

    @cached_property
    def candidate_has_multi_feature(self):
        return not isinstance(self.insts[0]['text'], str)

    @cached_property
    def feats(self):
        assert self.candidate_has_multi_feature
        feat2cis = defaultdict(list)
        for c_i, inst in enumerate(self.insts):
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
            all_text = [inst['text'] for inst in self.insts]
        all_text = [self.normalize_text(text) for text in all_text]

        to_embed = list({(self.cand_prefix + text) for text in all_text if (self.cand_prefix + text) not in text2emb})
        if to_embed:
            encoded = EvalSearcher.encode(self.model, self.tokenizer, to_embed, self.pooling_type, self.normalize)
            new_text2emb = {text: emb for text, emb in zip(to_embed, encoded)}
            text2emb |= new_text2emb
            io_util.write(self.cand_emb_path, text2emb)
            print(f'Saved {len(new_text2emb)} new candidate emb to {self.cand_emb_path}')
        return text2emb

    @cached_property
    def query2emb(self):
        text2emb = io_util.read(self.query_emb_path) if exists(self.query_emb_path) else {}
        all_text = [self.normalize_text(inst['query']) for inst in self.insts]

        to_embed = list({(self.query_prefix + text) for text in all_text if (self.query_prefix + text) not in text2emb})
        if to_embed:
            encoded = EvalSearcher.encode(self.model, self.tokenizer, to_embed, self.pooling_type, self.normalize)
            new_text2emb = {text: emb for text, emb in zip(to_embed, encoded)}
            text2emb |= new_text2emb
            io_util.write(self.query_emb_path, text2emb)
            print(f'Saved {len(new_text2emb)} new query emb to {self.query_emb_path}')
        return text2emb

    def get_query_emb(self, query):
        query = self.query_prefix + self.normalize_text(query)
        if query in self.query2emb:
            return self.query2emb[query]
        else:
            return EvalSearcher.encode(self.model, self.tokenizer, query, self.pooling_type, self.normalize)

    def get_caption_emb(self, caption):
        caption = self.cand_prefix + self.normalize_text(caption)
        if caption in self.cand2emb:
            return self.cand2emb[caption]
        else:
            return EvalSearcher.encode(self.model, self.tokenizer, caption, self.pooling_type, self.normalize)

    def match_by_dense(self):
        assert self.query2emb is not None and self.cand2emb is not None
        dists = []

        for inst in self.insts:
            query, caption, label = inst['query'], inst['text'], inst['label']

            if self.candidate_has_multi_feature:
                feat_dists = []
                for feat in caption:
                    diff = self.get_query_emb(self.normalize_text(query)) - self.get_caption_emb(self.normalize_text(feat))
                    feat_dists.append(np.linalg.norm(diff).item())
                dists.append(min(feat_dists))
            else:
                diff = self.get_query_emb(self.normalize_text(query)) - self.get_caption_emb(self.normalize_text(caption))
                dists.append(np.linalg.norm(diff).item())
        dists = [d ** 2 for d in dists]

        if self.do_rerank:
            assert not self.candidate_has_multi_feature
            pairs = [[self.normalize_text(inst['query']), self.normalize_text(inst['text'])] for inst in self.insts]
            rerank_scores = EvalSearcher.encode_pairs(self.reranker, self.reranker_tokenizer, pairs)
        else:
            rerank_scores = [None] * len(self.insts)

        for inst, dist, rerank_score in zip(self.insts, dists, rerank_scores):
            inst['mode'] = 'dense'
            inst['distance'] = dist
            inst['rerank_score'] = rerank_score
        return self.insts

    def match_by_exact(self):
        for inst in self.insts:
            inst['mode'] = 'exact'
            query_tokens = jieba.lcut(inst['query'])

            if not inst['text']:
                inst['distance'] = 100
            else:
                if self.candidate_has_multi_feature:
                    # inst['distance'] = 0.5 if all(any(feat.endswith(q_tok) for feat in inst['text']) for q_tok in query_tokens) else 2
                    inst['distance'] = 0.5 if all(q_tok in ' '.join(inst['text']) for q_tok in query_tokens) else 2
                else:
                    inst['distance'] = 0.5 if all((tok in inst['text']) for tok in query_tokens) else 2
        return self.insts


@dataclass
class Evaluator:

    save_dir: str
    dataset_path: str

    model_name: str
    pooling_type: str
    normalize: bool
    query_prefix: str
    cand_prefix: str

    query_threshold: float = None
    mode: str = 'dense'

    do_rerank: bool = False
    reranker_name: Optional[str] = None
    rerank_threshold: Optional[float] = None
    rerank_only_above: Optional[float] = None

    def __post_init__(self):
        assert self.mode in ('dense', 'exact')
        assert self.query_threshold is not None
        if not self.do_rerank:
            self.reranker_name = self.rerank_threshold = self.rerank_only_above = None

        self.searcher = Searcher(self.save_dir, self.dataset_path, self.model_name, self.pooling_type, self.normalize, self.query_prefix, self.cand_prefix,
                                 do_rerank=self.do_rerank, reranker_name=self.reranker_name)
        self.dataset_name = self.searcher.dataset_name
        self.model_alias = self.searcher.model_alias
        self.reranker_alias = self.searcher.reranker_alias

        if self.mode == 'dense':
            rerank = f'.rerank.{self.reranker_alias}' if self.do_rerank else ''
            self.result_path = join(self.save_dir, f'pair_results.{self.dataset_name}.{self.model_alias}{rerank}.json')
        else:
            self.result_path = join(self.save_dir, f'pair_results.{self.dataset_name}.exact.json')

    def get_results(self):
        # Do match
        insts = self.searcher.match_by_dense() if self.mode == 'dense' else self.searcher.match_by_exact()

        # Save
        io_util.write(self.result_path, insts)
        print(f'Saved {len(insts)} results to {self.result_path}')

        # Get metrics
        metric2score = self.get_metrics(insts, self.query_threshold, rerank_threshold=self.rerank_threshold, rerank_only_above=self.rerank_only_above)
        self.get_metrics_by_type(insts, self.query_threshold, rerank_threshold=self.rerank_threshold, rerank_only_above=self.rerank_only_above)
        return insts, metric2score

    @classmethod
    def get_metrics(cls, insts, threshold, rerank_threshold, rerank_only_above, do_print=True):
        threshold = threshold if threshold is not None else float('inf')
        preds = [int(inst['distance'] <= threshold and (rerank_threshold is None or (inst['rerank_score'] >= rerank_threshold or (rerank_only_above is not None and inst['distance'] <= rerank_only_above))))
                 for inst in insts]
        correct = [(pred == inst['label']) for pred, inst in zip(preds, insts)]
        acc = sum(correct) / len(correct) * 100
        metric2score = {f'overall accuracy ({len(correct)} pairs)': acc}

        pos_correct = [inst_correct for inst_correct, inst in zip(correct, insts) if inst['label'] == 1]
        neg_correct = [inst_correct for inst_correct, inst in zip(correct, insts) if inst['label'] == 0]
        if pos_correct:
            metric2score |= {f'pos-pair accuracy ({len(pos_correct)} pairs)': sum(pos_correct) / len(pos_correct) * 100}
        if neg_correct:
            metric2score |= {f'neg-pair accuracy ({len(neg_correct)} pairs)': sum(neg_correct) / len(neg_correct) * 100}

        if do_print:
            print('\nCorrect:\n')
            for is_correct, inst in zip(correct, insts):
                if is_correct:
                    print(inst)
            print('\nWrong:\n')
            for is_correct, inst in zip(correct, insts):
                if not is_correct:
                    print(('过召回: ' if inst['label'] == 0 else '漏召回: ') + str(inst))
            print()

        for metric, score in metric2score.items():
            print(f'{metric}: {score:.2f}')
        print()
        return metric2score

    @classmethod
    def get_metrics_by_type(cls, insts, threshold, rerank_threshold, rerank_only_above):
        type2insts = defaultdict(list)
        for inst in insts:
            if inst.get('type', None):
                type2insts[inst['type']].append(inst)
        if type2insts:
            print('=' * 30 + '\n')
            for type_, insts in type2insts.items():
                print(f'Metrics for type {type_} ({len(insts)} pairs)')
                cls.get_metrics(insts, threshold, rerank_threshold=rerank_threshold, rerank_only_above=rerank_only_above, do_print=True)

    @classmethod
    def ensemble(cls, paths, pooling='min'):
        """ Keep same logics as evaluate_ensemble.py """
        assert pooling in ('max', 'min', 'mean')
        path2insts = {path: io_util.read(path) for path, th in paths}
        path2th = {path: th for path, th in paths}
        path_ids = [{inst['id'] for inst in insts} for insts in path2insts.values()]
        # for p_i in range(1, len(paths)):
        #     assert path_ids[p_i] == path_ids[p_i - 1], 'Should have same insts'

        id2insts = defaultdict(list)
        for path, insts in path2insts.items():
            for inst in insts:
                if inst['distance'] > path2th[path]:
                    inst['distance'] = float('inf')
                else:
                    inst['distance'] = 0
                id2insts[inst['id']].append(inst)

        # Ensemble
        ensembled_insts = []
        for id_, insts in id2insts.items():
            distances = [inst['distance'] for inst in insts]
            if pooling == 'min':
                final_dist = min(distances)
            elif pooling == 'max':
                final_dist = max(distances)
            elif pooling == 'mean':
                final_dist = sum(distances) / len(distances)
            else:
                raise ValueError(pooling)
            ensembled_insts.append(insts[0] | {'distance': final_dist, 'mode': 'ensemble'})

        # Get metrics
        metric2score = cls.get_metrics(ensembled_insts, threshold=1, rerank_threshold=None, rerank_only_above=None)
        cls.get_metrics_by_type(ensembled_insts, threshold=1, rerank_threshold=None, rerank_only_above=None)
        return ensembled_insts, metric2score


def main():
    parser = ArgumentParser('Evaluate Pairs')
    parser.add_argument('--dataset', type=str, help='Dataset under ./dataset_pairs', required=True)
    parser.add_argument('--model', type=str, help='Model name or path', default='BAAI/bge-base-en-v1.5')
    parser.add_argument('--pooling', type=str, help='Encoder pooling style', default='cls', choices=['cls', 'mean'])
    parser.add_argument('--normalize', type=int, help='Whether normalize emb', default=1)
    parser.add_argument('--query_prefix', type=str, help='query_prefix', default='')
    parser.add_argument('--cand_prefix', type=str, help='cand_prefix', default='')
    parser.add_argument('--threshold', type=float, help='Search threshold', required=True)
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
        Evaluator.get_metrics(results, threshold=args.threshold, rerank_threshold=args.rerank_threshold, rerank_only_above=args.rerank_only_above)
    else:
        evaluator = Evaluator('evaluation', args.dataset, args.model, args.pooling, bool(args.normalize), args.query_prefix, args.cand_prefix,
                              query_threshold=args.threshold, mode=args.mode,
                              do_rerank=args.do_rerank, reranker_name=args.reranker_name, rerank_threshold=args.rerank_threshold, rerank_only_above=args.rerank_only_above)
        evaluator.get_results()


if __name__ == '__main__':
    main()
