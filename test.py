from run import Searcher
import io_util
from argparse import ArgumentParser


def main():
    parser = ArgumentParser('Evaluate')
    parser.add_argument('--data', type=str, help='Json path similar to test.json', default='test.json')
    parser.add_argument('--model', type=str, help='Model name or path', default='BAAI/bge-base-en-v1.5')
    parser.add_argument('--pooling', type=str, help='Encoder pooling style', default='cls', choices=['cls', 'mean', 'last'])
    parser.add_argument('--disable_normalization', help='Disable embedding normalization', action='store_true')
    parser.add_argument('--query_template', type=str, help='Prompt template for query', default=None)
    parser.add_argument('--candidate_template', type=str, help='Prompt template for candidate', default=None)
    parser.add_argument('--reranker_name', type=str, help='Reranker name or path', default=None)
    args = parser.parse_args()

    cases = io_util.read(args.data)
    searcher = Searcher(args.model, pooling_type=args.pooling, query_template=args.query_template, candidate_template=args.candidate_template,
                        reranker_name=args.reranker_name)
    print(f'Using model: {searcher.model_alias}')

    for case in cases:
        case['queries'] = [searcher.normalize_query(line) for line in case['queries']]
        case['candidates'] = [searcher.normalize_candidate(line) for line in case['candidates']]

    results = []
    for case in cases:
        q_emb = searcher.encode(searcher.model, searcher.tokenizer, case['queries'], searcher.pooling_type, searcher.normalize)
        c_emb = searcher.encode(searcher.model, searcher.tokenizer, case['candidates'], searcher.pooling_type, searcher.normalize)

        sim = q_emb @ c_emb.T
        dists = (2 - 2 * sim).flatten().tolist()  # Faiss distance: L2 square

        rank_scores = []
        if searcher.reranker_name:
            pairs = []
            for q in case['queries']:
                for c in case['candidates']:
                    pairs.append([q, c])
            rank_scores = searcher.encode_pairs(searcher.reranker, searcher.reranker_tokenizer, pairs)

        result = []
        for q in case['queries']:
            for c in case['candidates']:
                result.append({'query': q, 'candidate': c, 'dist': dists[len(result)],
                               'rank_score': None if not rank_scores else rank_scores[len(result)]})
        results.append(result)

    # Print
    for result in results:
        for line in result:
            if line['rank_score'] is not None:
                print(f'{line["dist"]:.4f} | {line["rank_score"]:.4f} | {line["query"]} -> {line["candidate"]}')
            else:
                print(f'{line["dist"]:.4f} | {line["query"]} -> {line["candidate"]}')
        print('=' * 20 + '\n')


if __name__ == '__main__':
    main()
