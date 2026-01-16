## Usage

### Docker
```bash
docker pull yuemingy/miles:dsv32-tilelang

docker run -it --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 --name miles_dsv32_tilelang yuemingy/miles:dsv32-tilelang /bin/zsh

git clone https://github.com/xiuhu17/miles
cd miles
git checkout tilelang
pip install -e .

# if shows Megatron does not support numpy 2.x
pip install numpy==1.26.4
```

### File to replace
File under the ```file_to_replace``` folder are the files used to replace in megatron. The target dir to replace is:

```Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa.py```

```Megatron-LM/megatron/core/transformer/tilelang_kernel/__init__.py```

```Megatron-LM/megatron/core/transformer/tilelang_kernel/sparse_mla_bwd.py```

```Megatron-LM/megatron/core/transformer/tilelang_kernel/sparse_mla_fwd.py```

```Megatron-LM/megatron/core/transformer/dot_product_attention_context_parallel.py```

```Megatron-LM/megatron/core/transformer/multi_latent_attention.py```

```Megatron-LM/megatron/core/transformer/transformer_config.py```

### Torch Native Fwd, Bwd for absorbption
For bwd, it is in: ```absorbtion/bwd.py```
For fwd,  use v from k ```v = k[..., : dim_v]```


### Quick test with 5 layer model
#### model download

```
hf download Pinaster/DeepSeek-V3.2-5layer --local-dir /root/models/DeepSeek-V3.2-5layer
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




