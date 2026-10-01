# SGLang rebase validation

Temporary dummy PR for running `run-ci-megatron` against the rebased SGLang candidate. Do not merge.

- SGLang candidate: `fa5e0b030b2cc4b3f260ee774b120fd88dbd88e4`.
- Candidate branch: `sgl-project/sglang:codex/upstream-sglang-miles-latest-passed-main`.
- Upstream baseline: `95047d3464764ccfa2ee3391e1c07e75df37c6aa`, validated by [Scheduled Full Run](https://github.com/sgl-project/sglang/actions/runs/36199886518).
- Set `ci-sglang-pr` in the PR body to the full candidate SHA so CPU and GPU jobs fetch the same revision.
