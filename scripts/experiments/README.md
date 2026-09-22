# Experiment queues

These are the queues that produced the numbers in `docs/RESULTS.md` outside
the generated sections, kept as the record of exactly what was run. They are
not entry points: several wait on another queue's log before starting, the way
they were chained at the time. To reproduce the published protocol, run
`scripts/run_all.sh`; to re-run one experiment, read its script and run the
training commands it contains.

Labels in `results/` follow `<preset>-<corpus>-s<seed>`. On MPS a seed does
not pin a run, so `s7`, `s17` and `s27` name three independent runs.

| script | what it trained | where it is reported |
|---|---|---|
| `queue_cosine.sh` | blink-tiny and blink-small on the synthetic corpus, cosine head, three runs each (the current headline models) | RESULTS §3, generated |
| `queue_wanli_sizes.sh` | blink-tiny and blink-small on WANLI, cosine head | RESULTS §3 *External corpora*, and the blink-small note after it |
| `queue_section3.sh` | the same set on the previous option head (format 2) | superseded by the two above; its containers are archived |
| `queue_cross.sh` | three runs with the question-to-state cross layer | RESULTS *`judgment` and question-to-state attention* (previous head) |
| `queue_seeds.sh` | the FiLM ablation, three runs per arm | RESULTS *The FiLM ablation* (previous head) |
| `queue_external.sh` | jevlike at widths 64 and 288 on Blink's corpus | RESULTS *Head to head with jevlike* (previous head) |
| `queue_wikispeedia.sh` | blink-tiny and jevlike on Wikispeedia next-click | RESULTS *Head to head with jevlike, on jevlike's ground* (previous head) |
| `queue_runs.sh` | the first single-run queue | superseded; kept for the history in RESULTS §6 |

"Previous head" sections compare arms trained under the same head, so their
comparisons stand, but their containers are format 2 and no longer load.
Re-running them on the current head is RESULTS §6 item 2. The jevlike and
Wikispeedia queues need a checkout of jevlike next to this one (`../jevlike`).
