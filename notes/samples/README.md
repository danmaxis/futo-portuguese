# Synthetic typo samples

Small excerpts of the synthetic typo corpora used to fine-tune the model.
The full pools (200K hand-rolled + Claude-generated batches) are reproducible
via `scripts/04a_build_wordfreq.py`, `scripts/lib_typo_synthesis.py`, and
`scripts/run_claude_synth_v8_1.sh`. We publish only samples here to (a) show
the data shape and (b) keep the repo small.

The corresponding *real* typo pool is NOT published — it is the maintainer's
own typed text and is gitignored.
