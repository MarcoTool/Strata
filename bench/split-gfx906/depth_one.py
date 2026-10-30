#!/usr/bin/env python3
"""One request per prompt length against a running Strata server; prints one TSV line each.

Usage: PORT=8080 [NGEN=300] python3 depth_one.py 16384 32768 65536 131072 196608 258048

Each prompt is prompts/ctx-N.txt (N tokens of Strata's own docs and sources, cut with the
Qwen3.8-Flash-Next tokenizer) plus a fixed question, with a new id at the start so the
engine cannot reuse a cached prefix.  Thinking is off, temperature 0, NGEN answer tokens.
The speeds are the server's own timings; wall_s is the whole request seen by this client.
For a split, the prefill card's read time is the server's "split:" log line, not prefill_tok_s.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

PORT = int(os.environ.get('PORT', '8080'))
NGEN = int(os.environ.get('NGEN', '300'))
MODEL = 'qwen3.8-flash-next-iq3_s'
PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts')
# the question appended to every prompt in the measurements, kept as it was sent
# ("Explain the content, structure and key points of the text above in detail, in at least 400 characters.")
QUESTION = '\n\n請詳細說明上文的內容、結構與重點，至少寫 400 字。'

print('depth\tprompt_n\tcache_n\tprefill_tok_s\tprefill_s\twall_s\tdec_tok_s\tdec_n\tdraft_acc', flush=True)
for tag in sys.argv[1:]:
    with open(os.path.join(PROMPTS, f'ctx-{tag}.txt'), encoding='utf-8') as f:
        text = f.read() + QUESTION
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': f'[{uuid.uuid4()}]\n' + text}], 'max_tokens': NGEN,
            'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=7200))
    except urllib.error.HTTPError as e:
        print(tag, 'error', e.code, e.read().decode(errors='ignore')[:300], flush=True)
        continue
    wall = time.time() - t0
    t = r.get('timings', {})
    dn, da = t.get('draft_n') or 0, t.get('draft_n_accepted') or 0
    print(f"{tag}\t{t.get('prompt_n')}\t{t.get('cache_n')}\t{t.get('prompt_per_second') or 0:.1f}\t"
          f"{(t.get('prompt_ms') or 0) / 1000:.1f}\t{wall:.1f}\t{t.get('predicted_per_second') or 0:.1f}\t"
          f"{t.get('predicted_n')}\t{da}/{dn}", flush=True)
