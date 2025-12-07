TASK="circle_packing_modular"
# TASK="first_autocorr_inequality"

CONFIG_POSTFIX="it_XL"


# # test command with verifier
OPENEVOLVE_CONFIG_PATH=$PWD/examples/${TASK}/configs/config_${TASK}_${CONFIG_POSTFIX}.yaml \
PYTHONPATH=$PWD \
python $PWD/examples/${TASK}/evaluators/evaluator_modular.py \
$PWD/examples/${TASK}/initial_programs/initial_program.py