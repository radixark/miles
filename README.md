<div align="center">

<!-- Yiping Wang, Shao-Rong Su, Zhiyuan Zeng, Eva Xu, Liliang Ren, Xinyu Yang, Zeyi Huang, Xuehai He, Luyao Ma, Baolin Peng, Hao Cheng, Pengcheng He, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen -->

# ThetaEvolve: Test-time Learning on Open Problems


[Yiping Wang](https://ypwang61.github.io/), 
[Shao-Rong Su](https://www.linkedin.com/in/andysu0731/), 
[Zhiyuan Zeng](https://zhiyuan-zeng.github.io/), 
[Eva Xu](https://www.linkedin.com/in/evaxu9187/),
[Liliang Ren](https://renll.github.io/), 
[Xinyu Yang](https://xinyuyang.me/),
[Zeyi Huang](https://oodbag.github.io/),
[Xuehai He](https://sheehan1230.github.io/), 
[Luyao Ma](https://www.linkedin.com/in/luyao-ma-4092a8273/),
[Baolin Peng](https://www.microsoft.com/en-us/research/people/baolinpeng/), 
[Hao Cheng](https://www.microsoft.com/en-us/research/people/chehao/), 
[Pengcheng He](https://www.linkedin.com/in/pengcheng-he-42163729/),
[Weizhu Chen](https://www.microsoft.com/en-us/research/people/wzchen/), 
[Shuohang Wang](https://www.microsoft.com/en-us/research/people/shuowa/), 
[Simon Shaolei Du*](https://simonshaoleidu.com/), 
[Yelong Shen*](https://www.linkedin.com/in/yelong-shen-84b0122b/)

<br>

[![paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.23473)
[![Code](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/ypwang61/ThetaEvolve)
<!-- [![📁_W&B_LOGS](https://img.shields.io/badge/📁_W&B_LOGS-fcd022?style=for-the-badge&logo=wandb&logoColor=000)](https://wandb.ai/yipingwanguw/verl_few_shot?nw=nwuseryipingwang22) -->
<!-- [![X_Summary](https://img.shields.io/badge/X_Summary-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/ypwang61/status/1917596101953348000) -->
<!-- [![Models/Dataset](https://img.shields.io/badge/Models/Dataset-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/ypwang61/one-shot-rlvr-6827f72c3359b2ffe75fc1a8) -->

</div>

## Outline

We introduce **ThetaEvolve**, an open-source pipeline that simplifies (e.g., with single LLM) and extends AlphaEvolve to efficiently scale both ❄️in-context learning and 🔥RL training at test time.

With ThetaEvolve, an 8B model can outperform AlphaEvolve on open optimization problems by scaling compute for inference or test-time RL🚀:

⭕Circle packing:

* AlphaEvolve (Gemini-2.0-Flash/Pro) : 2.63586276

* **Ours (R1-Qwen3-8B): 2.63598308**

![Figure1](assets/f0.png)




## Setup

Our RL environment follows the same setup as [miles](https://github.com/radixark/miles) and [OpenEvolve](https://github.com/codelion/openevolve). We use Docker (run in ThetaEvolve folder):

```bash
# fixed image, haven't checked on the latest image
docker pull radixark/miles:v0.5.0rc0-cu126

# Start the container
docker run --rm --name miles-evolve \
  --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace -w /workspace \
  -v /path/to/disk:/path/to/disk \
  -it radixark/miles:v0.5.0rc0-cu126 /bin/bash
```

After entering the docker, run the installation commands:

```bash
cd /workspace
pip install -e .
cd openevolve_adapted
pip install --ignore-installed blinker
rm -rf openevolve.egg-info && pip install -e .
cd ..
```

## Tasks
You could check our tasks in `openevolve_adapted/examples`. It is easy to extend to more tasks with continous objective values.

## Run
To run the experiments, you could change the parameters in `run.sh`, and then directly run `bash run.sh`

Fist, remember to set the save_path to store ckpts:

```
export SAVE_PATH=/path/to/disk/save
```

Then for example, if you want to run prorl-v2-1.5B, circle packing, RL training, original score as reward, you could set:

```bash
#### Model selection ####
SMALL_MODEL_NAME="dpsk_prorl_v2_1.5b"

#### Task configuration ####
TASK="circle_packing_modular"

#### CONFIG_POSTFIX options ####
CONFIG_POSTFIX="it_XL"

#### Training mode: True for training, False for inference-only ####
IS_TRAINING=True

#### Training parameters ####
# Options: "original_reward", "rl_normalized_reward"
REWARD_PROCESS_TYPE="original_reward"

#### Lazy output penalty ####
# 1 -> child = parent
# 2 -> child = any program in database
LAZY_OUTPUT_PENALTY=1
```

Finally set the wandb configurations:
```bash
WANDB_API_KEY=aaa
WANDB_ENTITY=bbb
WANDB_PROJECT=ccc
```


Then you can directly run
```bash
bash run.sh
```

You could also adjust more parameters in `scripts_evolve/Nemotron-Research-Reasoning-Qwen-1.5B/general.sh`. Like ckpt saving frequency (default 10), number of evaluation threads (default 16), gpus (default 8), etc.


## Results
Some results we obtain are available in `Results`. You can run `python vis.py` to see the verification results in each sub-task directory.

For example, we have our best-known solution for circle packing (with zero tolerance) in `Results/CirclePacking/figs/8B-w_RL@65-Formal.png` and AlphaEvolve's solution in `Results/CirclePacking/figs/AlphaEvolve.png`:

<div align="center">
<img src="Results/CirclePacking/figs/8B-w_RL@65-Formal.png" width="49%">
<img src="Results/CirclePacking/figs/AlphaEvolve.png" width="47%">
</div>

We point out that our solution is better than AlphaEvolve’s, and that our configuration is asymmetric, whereas AlphaEvolve’s solution is symmetric.



The program for finding it (with 1e-6 tolerance as OpenEvolve verification, detailed in paper) is shown in `Results/CirclePacking/programs/8B-w_RL@65.py`. We also provide results from other tasks for visualization.

## Citation
If you find our work useful, please consider citing:

```bibtex
@article{wang2025thetaevolve,
  title={ThetaEvolve: Test-time Learning on Open Problems},
  author={Wang, Yiping and Su, Shao-Rong and Zeng, Zhiyuan and Xu, Eva and Ren, Liliang and Yang, Xinyu and Huang, Zeyi and He, Xuehai and Ma, Luyao and Peng, Baolin and Cheng, Hao and He, Pengcheng and Chen, Weizhu and Wang, Shuohang and Du, Simon Shaolei and Shen, Yelong},
  journal={arXiv preprint 2511.23473},
  year={2025}
}

```
