# Terminal Universe async training

This run-specific recipe uses 338 audited tasks and their prebuilt E2B templates.
It reserves one 8-GPU trainer node and three TP8 rollout nodes. Configure endpoint
credentials through environment variables; never commit them with the recipe.

Defaults: 256 concurrent episodes, 8 prompts with 16 samples each, 1,000 updates,
learning rate 1e-6, and checkpoints every 50 updates. R3 is enabled and generation
pauses in place. Terminus 2 summarization and linear-history recording are enabled.
The model context is 65,536 tokens, with 16,384 output tokens per model call.

Extract the published dataset archive rather than copying expanded S3 objects,
so executable and directory modes are preserved. Verify the archive and template
map against the release checksums recorded in the run manifest before launching.
Keep artifacts on a sufficiently large node-local scratch filesystem.
