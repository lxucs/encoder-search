from run import main_args, Evaluator


def get_prec_at_recall(target_recall, max_threshold=1.7):
    args = main_args()
    args.threshold = max_threshold
    min_th_exceed_target, max_th_below_target = float('inf'), float('-inf')
    runs = []
    recall, step = None, -0.1
    min_step = 0.002

    while not recall or abs(recall - target_recall) > 0.2:
        print(f'Threshold: {args.threshold:.2f}')
        evaluator = Evaluator('evaluation', args.dataset, args.gold_score, args.mode,
                              args.model, args.pooling, not args.disable_normalization, args.query_template, args.candidate_template,
                              is_colbert=args.is_colbert, use_simple_colbert_query=args.use_simple_colbert_query, use_colbert_linear=not args.disable_colbert_linear,
                              query_threshold=args.threshold, topk=args.topk,
                              do_rerank=args.do_rerank, reranker_name=args.reranker_name, rerank_threshold=args.rerank_threshold, rerank_only_above=args.rerank_only_above)
        _, ds2metric2score = evaluator.get_results(save_results=False)
        recall = ds2metric2score[args.dataset]['query_recall']
        precision = ds2metric2score[args.dataset]['query_precision']
        runs.append((recall, precision, args.threshold))

        if recall > target_recall:
            min_th_exceed_target = min(min_th_exceed_target, args.threshold)
        else:
            max_th_below_target = max(max_th_below_target, args.threshold)
        print(f'Threshold range: {max_th_below_target:.4f} to {min_th_exceed_target:.4f}')

        if recall > target_recall and step > 0:  # to decrease th
            step = -(step - min_step)
        elif recall < target_recall and step < 0:  # to increase th
            step = -step - min_step
        args.threshold += step

        if args.threshold > min_th_exceed_target:
            actual_threshold = min_th_exceed_target - min_step
            step = actual_threshold - args.threshold
            args.threshold = actual_threshold
        elif args.threshold < max_th_below_target:
            actual_threshold = max_th_below_target + min_step
            step = actual_threshold - args.threshold
            args.threshold = actual_threshold
        print(f'=' * 60 + '\n\n')

    runs = sorted(runs, reverse=True)
    target_run = None
    for run in runs:
        if not target_run and abs(run[0] - target_recall) <= 0.2:
            target_run = run
        print(f'Recall: {run[0]:.2f} | Precsion: {run[1]:.2f} (threshold={run[-1]:.4f})')

    run = target_run
    print(f'\nTarget:')
    print(f'Recall: {run[0]:.2f} | Precsion: {run[1]:.2f} (threshold={run[-1]:.4f})')


if __name__ == '__main__':
    get_prec_at_recall(target_recall=90, max_threshold=1.7)
