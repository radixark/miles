### Prepare environment
#### Docker settings
```bash
# 1. create a docker network
docker network create swe-net

# 2. create environment docker
docker run -itd \
  --name swe_env \
  --shm-size 16g \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /mnt/data:/data \
  -v /home/sglang-rl/<your_name>:/workspace \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network swe-net \
  ubuntu:latest \
  /bin/bash

# 3. create miles docker
docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /mnt/data/cache/huggingface:/root/.cache/huggingface \
  -v /mnt/data:/data \
  -v /home/sglang-rl/<your_name>:/workspace \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --network swe-net \
  --name miles_<your_name> \
  slimerl/slime:latest \
  /bin/zsh

docker exec -it miles_<your_name> /bin/bash

apt update && apt install -y zsh curl git python3 python3-pip docker.io
```
note: `-v /var/run/docker.sock:/var/run/docker.sock` is required for Docker-in-Docker SWE environment execution; use `--network swe-net` to enable communication between training & environment.

#### Installation

In **environment docker**, install Gym
```bash
git clone https://github.com/yueming-yuan/Gym
cd Gym

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev --group docs

# configure env.yaml
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: your-openai-api-key
policy_model_name: gpt-4.1-2025-04-14
default_host: 0.0.0.0" > env.yaml
```
note: set host IP to `0.0.0.0` to enable commnunications between dockers.

then set up for SWE-agent server:
```bash
cd responses_api_agents/mini_swe_agent
uv pip install -r requirements.txt
```
Now you should be able to run the SWE-agent server.

For **miles docker** setup, please follow the standard setup process.

### Preparing data
In **miles docker**, download **SWE-Gym** data from huggingface and convert it to Miles' prompt data format with this script.
```
cd miles/examples/swe-agent
python download_and_process_data.py --input SWE-Gym/SWE-Gym --output root/swe_train.jsonl
```

### Running train
1. In environment docker, launch the agent server
```bash
cd responses_api_agents/mini_swe_agent
./start_server.sh
```


2. In miles docker,
(1) export `SWE_AGENT_GYM_URL` to be the port of the second server you started in Gym in environment docker, whose `server_type` is `responses_api_agents`. `swe_env` is the environment docker's name; replace it if you changed the name.
(minor TODO: modify the port selections to avoid setting this every time.) (2) launch the training.
```bash
export SWE_AGENT_GYM_URL="http://swe_env:<port_of_responses_api_agents>"
bash examples/swe-agent/run-qwen3-4b-mis.sh
```


### Troubleshooting
1. The first time of every SWE environment can be slow, and may need to wait before generation, because each SWE-Gym task has a specific docker, and `docker pull` takes time.
1. Sometimes the environment may also be slow at evaluation. The timeout of evaluation is 10 minutes by default. If the server is stuck at `[EVAL]<instance> Running eval`, you may need to wait for it.