# Task 5.18 — first-divergence token logp / rank

Same 5.17 post-submit prefixes and gold/call token IDs. No training, no full-episode generation. Models: base + 5.16 `step_3`.

At the first index where gold final and lookup_order continuations differ, score `logp(gold_token) − logp(call_token)` and 1-based ranks on `prefix + shared continuation prefix`.

## Table

| example | role | model | div@ | gold id/text | gold logp | gold rank | call id/text | call logp | call rank | Δlogp (gold−call) | top1 id/text |
|---|---|---|---:|---|---:|---:|---|---:|---:|---:|---|
| d1-g101 | fail | base | 0 | 26501/'REF' | -1.616526 | 2 | 73/'[' | -0.444651 | 1 | -1.171875 | 73/'[' (=call) |
| d1-g101 | fail | step_3 | 0 | 26501/'REF' | -1.481955 | 2 | 73/'[' | -0.497580 | 1 | -0.984375 | 73/'[' (=call) |
| d2-g101 | fail | base | 0 | 26501/'REF' | -1.730014 | 2 | 73/'[' | -0.401889 | 1 | -1.328125 | 73/'[' (=call) |
| d2-g101 | fail | step_3 | 0 | 26501/'REF' | -1.647178 | 2 | 73/'[' | -0.428428 | 1 | -1.218750 | 73/'[' (=call) |
| d0-g100 | control | base | 0 | 26501/'REF' | -0.213278 | 1 | 73/'[' | -3.338278 | 3 | 3.125000 | 26501/'REF' (=gold) |
| d0-g100 | control | step_3 | 0 | 26501/'REF' | -0.193458 | 1 | 73/'[' | -3.482521 | 3 | 3.289062 | 26501/'REF' (=gold) |
| d3-g100 | control | base | 0 | 26501/'REF' | -0.275391 | 1 | 73/'[' | -2.814454 | 2 | 2.539062 | 26501/'REF' (=gold) |
| d3-g100 | control | step_3 | 0 | 26501/'REF' | -0.249072 | 1 | 73/'[' | -2.975635 | 2 | 2.726562 | 26501/'REF' (=gold) |

