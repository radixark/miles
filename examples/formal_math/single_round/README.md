# Usage

For the minimal demo:

```shell
# install dependencies
apt update && apt install -y docker-cli
pip install kimina-client

# prepare data
python examples/formal_math/single_round/prepare_data.py --output-name minimal_demo

# run
examples/formal_math/single_round/run_minimal.py
```

The code also support more complicated cases, e.g.:

* SFT + RL
* Data filter + RL
