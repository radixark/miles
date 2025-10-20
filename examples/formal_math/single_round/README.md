# Usage

For the minimal demo:

```shell
# install dependencies
apt update && apt install -y docker-cli
pip install kimina-client polars

# prepare data
python examples/formal_math/single_round/prepare_data.py --output-name minimal_demo

# prepare ray, model, test dataset, etc
# normally just use this script, but here we want to demonstrate run_minimal.py, thus skip ray-submit part
MILES_SCRIPT_ENABLE_RAY_SUBMIT=0 python examples/formal_math/single_round/run.py

# run
python examples/formal_math/single_round/run_minimal.py
```

As a reference, the docker env may be started as:

```shell
export WANDB_API_KEY=...

docker pull slimerl/slime:latest
docker stop miles_adhoc_0 || true
docker rm miles_adhoc_0 || true
docker run \
    -d \
    --name miles_adhoc_0 \
    --gpus all \
    --ipc=host \
    --network host \
    --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --workdir /host_home/primary_synced/miles \
    -v ~/.ssh:/root/.ssh \
    -v /data/tom:/host_home \
    -v /scratch/.cache:/root/.cache \
    -v /data/miles_ci:/data/miles_ci \
    -v /scratch:/host_local_disk \
    -v /data/miles_ci/models:/root/models \
    -v /data/miles_ci/datasets:/root/datasets \
    -v /data/tom/miles_shared_data:/root/shared_data \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -e WANDB_API_KEY=${WANDB_API_KEY} \
    slimerl/slime:latest \
    /bin/bash -c 'echo now sleep forever ; while true; do sleep 2; done'
docker exec miles_adhoc_0 \
    /bin/bash -c \
    "pip install -e . && apt update && apt install -y docker-cli && pip install kimina-client polars"
docker exec -it miles_adhoc_0 /bin/zsh
```

The code also support more complicated cases, e.g.:

* SFT + RL
* Data filter + RL
