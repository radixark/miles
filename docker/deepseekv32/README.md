## Usage

### Docker
```bash
docker pull yueming11/miles:dsv32-dev

docker run --gpus all --ipc=host --shm-size=16g   --ulimit memlock=-1 --ulimit stack=67108864 --name miles_dsv32 yueming11/miles:dsv32-dev /bin/zsh

git clone https://github.com/radixark/miles.git
git checkout dsv32
cd dsv32
pip install -e .

# if shows Megatron does not support numpy 2.x
pip install numpy==1.26.4
```

### Quick test with 5 layer model
#### model download

```
hf download Pinaster/DeepSeek-V3.2-5layer /root/models/DeepSeek-V3.2-5layer
```

#### Prepare model for training
Note: need to change the paths, for all commands below see `scripts/run_deepseek_v32.py` for details

Step 1. download dataset & convert fp8 hf checkpoint to bf16 with one node
```
python scripts/run_deepseek_v32.py prepare-single --model-name DeepSeek-V3.2-5layer --megatron-model-type deepseek-v32-5layer
```

Step 2. convert hf checkpoint to megatron checkpoint with multiple nodes
```
python scripts/run_deepseek_v32.py prepare-spmd --model-name DeepSeek-V3.2-5layer --megatron-model-type deepseek-v32-5layer
```

#### Launch training
```
python scripts/run_deepseek_v32.py train --model-name DeepSeek-V3.2-5layer --megatron-model-type deepseek-v32-5layer
```