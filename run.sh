


########################### CONFIGURATION SECTION - EDIT THESE VARIABLES #############################

#### Important: replace SAVE_PATH with your path with enough space ####
export SAVE_PATH=/path/to/disk

#### Model selection ####
SMALL_MODEL_NAME="dpsk_prorl_v2_1.5b"
# SMALL_MODEL_NAME="dpsk_distill_qwen3_8b"

#### Task configuration ####
# TASK="hadamard_matrix"
TASK="circle_packing_modular"
# TASK="third_autocorr_inequality"
# TASK="second_autocorr_inequality"
# TASK="first_autocorr_inequality"

#### CONFIG_POSTFIX options ####
CONFIG_POSTFIX="it_XL"

#### Training mode: True for training, False for inference-only ####
# IS_TRAINING=True
IS_TRAINING=True

#### Training parameters ####
# Options: "original_reward", "rl_normalized_reward"
REWARD_PROCESS_TYPE="original_reward"

#### Lazy output penalty ####
# 1 -> child = parent
# 2 -> child = any program in database
LAZY_OUTPUT_PENALTY=1

#### Random seed ####
SEED=3407

#### Different initial program ####
INITIAL_PROGRAM_POSTFIX=""
# INITIAL_PROGRAM_POSTFIX="_sota"

#### Additional note for file names ####
NOTE=""

#### Replace with your own wandb settings ####
WANDB_API_KEY=aaa
WANDB_ENTITY=bbb
WANDB_PROJECT=ccc

# ########################## END CONFIGURATION SECTION #############################



POSTFIX_STR="_seed${SEED}${INITIAL_PROGRAM_POSTFIX}${NOTE}"


if [ "$SMALL_MODEL_NAME" = "dpsk_prorl_v2_1.5b" ]; then
    MODEL_FAMILY="nvidia"
    MODEL_NAME="Nemotron-Research-Reasoning-Qwen-1.5B"
    models_file_name="deepseek-r1-distill-qwen-1.5B.sh"
elif [ "$SMALL_MODEL_NAME" = "dpsk_distill_qwen3_8b" ]; then
    MODEL_FAMILY="deepseek-ai"
    MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B"
    models_file_name="qwen3-8B.sh"
else
    echo "Unknown SMALL_MODEL_NAME: $SMALL_MODEL_NAME"
    exit 1
fi
echo "Using model: $MODEL_NAME"


######################################################

# Path configuration
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tmp
mkdir -p $SAVE_PATH/hf
mkdir -p $SAVE_PATH/wandb
mkdir -p $SAVE_PATH/shm
mkdir -p $SAVE_PATH/triton


# setup paths
export TMPDIR=/tmp # have to be real tmp path
export HF_HOME=$SAVE_PATH/hf
export HUGGINGFACE_HUB_CACHE=$SAVE_PATH/hf/hub
export TRANSFORMERS_CACHE=$SAVE_PATH/hf/hub
export HF_DATASETS_CACHE=$SAVE_PATH/hf/datasets
export SAVE_SHM_DIR=$SAVE_PATH/shm
export TRITON_CACHE_DIR=$SAVE_PATH/triton

# wandb
export WANDB_CACHE_DIR=$SAVE_PATH/wandb
export WANDB_DIR=$SAVE_PATH/wandb
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_ENTITY=$WANDB_ENTITY
export WANDB_PROJECT=$WANDB_PROJECT

# ########################## AUTO-GENERATED PATHS #############################

# Auto-generated paths using variables
INITIAL_PROGRAM="openevolve_adapted/examples/${TASK}/initial_programs/initial_program${INITIAL_PROGRAM_POSTFIX}.py"
EVALUATOR_FILE="openevolve_adapted/examples/${TASK}/evaluators/evaluator_modular.py"
CONFIG_YAML="openevolve_adapted/examples/${TASK}/configs/config_${TASK}_${CONFIG_POSTFIX}.yaml"

# Reward suffix mapping
case "$REWARD_PROCESS_TYPE" in
    "original_reward") REWARD_SUFFIX="" ;;
    "rl_normalized_reward") REWARD_SUFFIX="_rlnorm" ;;
    *) REWARD_SUFFIX="_${REWARD_PROCESS_TYPE}" ;;
esac

# Generate RUN_NAME
RUN_NAME="${SMALL_MODEL_NAME}_tr${IS_TRAINING}_l${LAZY_OUTPUT_PENALTY}_${TASK}_${CONFIG_POSTFIX}${REWARD_SUFFIX}${POSTFIX_STR}"

# ########################## MODEL SETUP #############################

FORCE_DOWNLOAD=0  # set to 1 to force re-download
# Check if model already exists and is complete
if [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ] && [ -f "$SAVE_SHM_DIR/$MODEL_NAME/config.json" ] && [ $FORCE_DOWNLOAD -eq 0 ]; then
    echo "Model $MODEL_NAME already exists at $SAVE_SHM_DIR/$MODEL_NAME, skipping download"
else
    # delete the original directory if it exists
    if [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ]; then
        echo "Incomplete model directory found at $SAVE_SHM_DIR/$MODEL_NAME, deleting and re-downloading"
        rm -rf "$SAVE_SHM_DIR/$MODEL_NAME"
    fi
    echo "Downloading model $MODEL_NAME to current directory first..."
    hf download $MODEL_FAMILY/$MODEL_NAME --local-dir ./$MODEL_NAME

    # Create target directory if it doesn't exist
    mkdir -p $SAVE_SHM_DIR

    # Move the downloaded model to target location
    echo "copy model from ./$MODEL_NAME to $SAVE_SHM_DIR/$MODEL_NAME"
    cp -r $MODEL_NAME $SAVE_SHM_DIR/

    echo "Model download and move completed"
fi

source scripts/models/${models_file_name}
if [ ! -d "$SAVE_SHM_DIR/${MODEL_NAME}_torch_dist" ] || [ $FORCE_DOWNLOAD -eq 1 ]; then
    echo "Converting HF model to torch dist format..."
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} --hf-checkpoint $SAVE_SHM_DIR/$MODEL_NAME --save $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist
    echo "Conversion completed, torch dist model saved at $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist"
else
    echo "Torch dist model already exists at $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist, skipping conversion"
fi

# ########################## MAIN EXECUTION #############################


echo "=== Experiment Configuration ==="
echo "RUN_NAME: ${RUN_NAME}"
echo "TASK: ${TASK}"
echo "INITIAL_PROGRAM: ${INITIAL_PROGRAM}"
echo "EVALUATOR_FILE: ${EVALUATOR_FILE}"
echo "CONFIG_YAML: ${CONFIG_YAML}"
echo "SAVE_PATH: ${SAVE_PATH}"
echo "================================"

# you can replace general.sh with general_debug.sh for debug mode
# General script with configurable parameters
bash scripts_evolve/${MODEL_NAME}/general.sh \
    "${WANDB_PROJECT}" \
    "${RUN_NAME}" \
    "${INITIAL_PROGRAM}" \
    "${EVALUATOR_FILE}" \
    "${CONFIG_YAML}" \
    "${SAVE_PATH}" \
    "${IS_TRAINING}" \
    "${LAZY_OUTPUT_PENALTY}" \
    "${REWARD_PROCESS_TYPE}" \
    "${SEED}" \
    2>&1 | tee -a "${SAVE_PATH}/${RUN_NAME}/train_log.txt"