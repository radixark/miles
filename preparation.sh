export SAVE_PATH=/path/to/disk

################## DOCKER ##################

sudo docker pull radixark/miles:v0.5.0rc0-cu126


# clean test
sudo docker run --rm --name miles-evolve2 \
  --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace -w /workspace \
  -v $SAVE_PATH:$SAVE_PATH \
  -it radixark/miles:v0.5.0rc0-cu126 /bin/bash


############# INSTALL #############
cd /workspace
pip install -e .
cd openevolve_adapted
pip install --ignore-installed blinker
rm -rf openevolve.egg-info && pip install -e .
cd ..