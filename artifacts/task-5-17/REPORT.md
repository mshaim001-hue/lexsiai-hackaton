# Task 5.17 — logprob gold final vs post-submit tool call

No training, no full-episode generation. Teacher-forced mean token logprob on exact 5.14 post-submit prefixes.

- fails: `['d1-g101', 'd2-g101']` (call tokens = saved `new_token_ids`)
- controls: `['d0-g100', 'd3-g100']` (gold = saved success tokens; call = encoded `lookup_order` for that order)
- models: base + 5.16 `step_1`…`step_3`

## Per example

### d1-g101 (fail)

| model | mean_logp gold | mean_logp call | gap (gold−call) | Δgap vs base |
|---|---:|---:|---:|---:|
| base | -0.117615 | -0.034771 | -0.082844 |  |
| step_1 | -0.116051 | -0.034850 | -0.081201 | 0.001643 |
| step_2 | -0.113565 | -0.035579 | -0.077986 | 0.004858 |
| step_3 | -0.108050 | -0.036575 | -0.071475 | 0.011369 |

### d2-g101 (fail)

| model | mean_logp gold | mean_logp call | gap (gold−call) | Δgap vs base |
|---|---:|---:|---:|---:|
| base | -0.126371 | -0.034019 | -0.092353 |  |
| step_1 | -0.125492 | -0.034082 | -0.091409 | 0.000943 |
| step_2 | -0.122962 | -0.034522 | -0.088441 | 0.003912 |
| step_3 | -0.120406 | -0.035020 | -0.085386 | 0.006967 |

### d0-g100 (control)

| model | mean_logp gold | mean_logp call | gap (gold−call) | Δgap vs base |
|---|---:|---:|---:|---:|
| base | -0.020114 | -0.151258 | 0.131145 |  |
| step_1 | -0.019724 | -0.152522 | 0.132798 | 0.001653 |
| step_2 | -0.019316 | -0.153848 | 0.134532 | 0.003387 |
| step_3 | -0.018611 | -0.156927 | 0.138316 | 0.007171 |

### d3-g100 (control)

| model | mean_logp gold | mean_logp call | gap (gold−call) | Δgap vs base |
|---|---:|---:|---:|---:|
| base | -0.023463 | -0.130021 | 0.106558 |  |
| step_1 | -0.023057 | -0.131348 | 0.108291 | 0.001733 |
| step_2 | -0.022222 | -0.133769 | 0.111547 | 0.004989 |
| step_3 | -0.021547 | -0.136430 | 0.114883 | 0.008325 |

