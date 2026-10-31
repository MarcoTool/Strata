#!/usr/bin/env python3
"""The standard benchmark's long-prompt case, extended past 16K: 32K, 64K, 128K, 192K and 258K.

Usage: PYTHONPATH=<strata checkout> python3 long_standard.py --url http://127.0.0.1:8080 --out DIR [--reps 3]

`strata bench run` (docs/BENCHMARK_PROTOCOL.md) stops at 16K.  This sends the same kind of request at longer
lengths: the bundle's record lines ("record000001: alpha beta ... theta."), its summary question, and the
request settings of strata_cli.bench_standard itself (payload_of, stream_chat, record_of: thinking on, sampling,
512 output tokens, seed = repetition).  The record count per length comes from the bundle's 4K and 16K fixtures
(19.0 tokens per record).  Not the standard protocol: its output is not a standard report.
Writes DIR/records.jsonl (one record per request, the runner's own format) and prints one TSV line per request.
"""
import argparse
import json
import os
import sys

from strata_cli import bench_standard as bs

TARGETS = {'prompt-32k': 32768, 'prompt-64k': 65536, 'prompt-128k': 131072, 'prompt-192k': 196608, 'prompt-258k': 258048}


def bundle_constants():
    """Record format, question and tokens per record, from the bundle's build.json and its 4K/16K fixtures."""
    d = bs.corpus_dir()
    build = json.load(open(d / 'build.json', encoding='utf-8'))
    req = {n: json.load(open(d / 'requests' / f'performance-{n}-r1.json', encoding='utf-8')) for n in ('prompt-4k', 'prompt-16k')}
    r4, r16 = (build['record_counts'][n]['records'] for n in ('prompt-4k', 'prompt-16k'))
    t4, t16 = req['prompt-4k']['input_tokens_frozen'], req['prompt-16k']['input_tokens_frozen']
    per = (t16 - t4) / (r16 - r4)
    return build['record_format'], build['summary_question'], per, t4 - r4 * per


def spec_of(case, target, rep, fmt, question, per, overhead):
    rid = f'performance-{case}-r{rep}'
    n = round((target - overhead) / per)
    content = f'[case {rid}]\n' + '\n'.join(fmt.format(i) for i in range(1, n + 1)) + '\n' + question
    return {'id': rid, 'phase': 'performance', 'case_id': case, 'repetition': rep, 'round': None, 'seed': rep,
            'max_output_tokens': 512, 'concurrency': 1, 'category': None, 'target_input_tokens': target,
            'messages': [{'role': 'user', 'content': content}], 'tools': None, 'tool_choice': None, 'expected': None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--cases', default=','.join(TARGETS))
    ap.add_argument('--timeout', type=float, default=3600)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fmt, question, per, overhead = bundle_constants()
    target = bs.Target(a.url, os.environ.get('STRATA_API_KEY'))
    props = target.get('/props', timeout=15) or {}
    model = props.get('model_alias')
    print('case\trep\tinput_tokens\tcached\tprompt_tps\tdecode_tps\tdecode_tokens\tttft_s\ttotal_s\tstatus\tlength_ok', flush=True)
    with open(os.path.join(a.out, 'records.jsonl'), 'a', encoding='utf-8') as f:
        for case in a.cases.split(','):
            for rep in range(1, a.reps + 1):
                spec = spec_of(case, TARGETS[case], rep, fmt, question, per, overhead)
                raw = bs.stream_chat(target, spec, model, timeout_s=a.timeout)
                rec, _, extras = bs.record_of(spec, raw, {'max_shared_prefix_tokens': 0, 'input_tokens': None})
                rec['extras'] = extras
                f.write(json.dumps(rec) + '\n')
                f.flush()
                t, m, tm = rec['tokens'], rec['metrics'], rec['timings']
                print(f"{case}\t{rep}\t{t['input_tokens']}\t{t['cached_tokens']}\t{m['prompt_tps']}\t{m['decode_tps']}\t"
                      f"{t['decode_tokens']}\t{tm['ttft_s']}\t{tm['total_s']}\t{rec['status']}\t"
                      f"{not extras['length_outside_tolerance']}", flush=True)
                if rec['status'] == 'failed':
                    print(f"failed: {rec['error']}", file=sys.stderr, flush=True)
    print('LONG-DONE', flush=True)


if __name__ == '__main__':
    main()
