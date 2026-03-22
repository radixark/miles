# Docker release rule

We will publish 2 kinds of docker images:
1. stable version, which based on official sglang release. We will store the patch on those versions.
2. latest version, which aligns to `lmsysorg/sglang:latest`.

current stable version is:
- sglang v0.5.7 nightly-dev-20260103-24c91001 (24c91001cf99ba642be791e099d358f4dfe955f5), megatron dev 3714d81d418c9f1bca4594fc35f9e8289f652862

history versions:
- (patches removed — see git history for details)

The command to build:

```bash
just release
```

Before each update, we will test the following models with 64xH100:

- Qwen3-4B sync
- Qwen3-4B async
- Qwen3-30B-A3B sync
- Qwen3-30B-A3B fp8 sync
- GLM-4.5-355B-A32B sync
