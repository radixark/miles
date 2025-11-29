

# Setup on host machine.

docker network create skills-net

# Setup miles docker

docker run \
    -itd \
    --shm-size 32g \
    --gpus all \
    -v /data/.cache:/root/.cache \
    -v /dev/shm:/shm \
    -v /data/jiajun:/root/shared \
    --ipc=host \
    --privileged \
    --network skills-net \
    --name <slime container name> \
    slimerl/slime:latest \
    /bin/bash

# Setup Skills docker.

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


# In Skills docker.

git clone -b miles_skills https://github.com/guapisolo/miles.git /opt/miles
git clone -b miles https://github.com/guapisolo/Skills.git /opt/Skills

cd /opt/Skills
pip install -e .

# Install datasets.

cd nemo_skills/dataset
python3 aime25/prepare.py
python3 hle/prepare.py
python3 arena-hard/prepare.py

# Start skills server.
cd /opt/miles
python examples/eval/skills/skills_server.py \
  --host 0.0.0.0 \
  --port 9050 \
  --output-root /root/shared/skills-eval \
  --config-dir examples/eval/skills \
  --cluster local_cluster \
  --server-type openai

