# Docs

## Prerequisites
- A writable host directory for cached data (`/data/.cache`)
- Choose descriptive container names to replace the placeholders (`<miles container name>`, `<env container name>`).

## 1) Prepare host network
```bash
docker network create skills-net
```

## 2) Launch the miles container
```bash
docker run \
  -itd \
  --shm-size 32g \
  --gpus all \
  -v /data/.cache:/root/.cache \
  -v /dev/shm:/shm \
  --ipc=host \
  --privileged \
  --network skills-net \
  --name <miles container name> \
  radixark/miles:latest \
  /bin/bash
```

## 3) Launch the Skills container
```bash
docker run \
  -itd \
  --shm-size 32g \
  --gpus all \
  -v /data/.cache:/root/.cache \
  -v /dev/shm:/shm \
  --ipc=host \
  --privileged \
  --network skills-net \
  --name <env container name> \
  --network-alias skills_server \
  guapisolo/nemoskills:0.7.1 \
  /bin/bash
```

## 4) Inside the Skills container
Clone repos and install the Skills package:
```bash
git clone -b miles_skills https://github.com/guapisolo/miles.git /opt/miles
git clone -b miles https://github.com/guapisolo/Skills.git /opt/Skills

cd /opt/Skills
pip install -e .
```

Download/prepare datasets:
```bash
cd /opt/Skills/nemo_skills/dataset
python3 aime25/prepare.py
python3 hle/prepare.py
python3 arena-hard/prepare.py
```

Start the skills server:
```bash
cd /opt/miles
python examples/eval/nemo_skills/skills_server.py \
  --host 0.0.0.0 \
  --port 9050 \
  --output-root /opt/skills-eval \
  --config-dir examples/eval/nemo_skills/config \
  --cluster local_cluster \
  --max-concurrent-requests 512 \
  --openai-model-name miles-openai-model
```

You can now connect to the server at `skills_server:9050` from within the `skills-net` Docker network. The server always proxies evaluation traffic to an OpenAI-compatible sglang router (Miles starts and manage the router), so adjust `--openai-model-name` and `--max-concurrent-requests` as needed for your deployment.

The example scripts are in `examples/eval/scripts`. To run them, you need to follow the following steps (take Qwen3-4B as an example):

1. Inside the miles container, run the following command:

```bash
cd /root/miles
git pull
pip install -e .

# Download model weights (Qwen3-4B)
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

# Download training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# Convert Qwen3-4B model to torch dist
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist

# Run the script
bash examples/eval/scripts/run-qwen3-4B.sh
```
