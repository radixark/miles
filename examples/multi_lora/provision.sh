#!/bin/bash

# Download model weights (Qwen3-4B)
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

# Download training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# Download training dataset (gsm8k)
hf download --repo-type dataset zhuzilin/gsm8k \
  --local-dir /root/gsm8k
