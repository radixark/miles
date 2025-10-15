Commands

```shell
# install dependencies
pip install kimina-client

# prepare data
python examples/formal_math/single_round/prepare_data.py --dir-output-base /root/datasets/formal_math_single_round/

# run
MILES_DATASET_TRANSFORM_ID=... python examples/formal_math/single_round/run.py
```
