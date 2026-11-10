"""Version-independent same-node weight buckets for a separate serving environment."""

from miles.backends.lmdeploy_utils.tensor_buckets import serialize_tensor_bucket
from miles.backends.training_utils.weight_update.protocols.cuda_ipc import UpdateWeightFromTensor


class LMDeployWeightFromTensor(UpdateWeightFromTensor):
    """Reuse placement and barriers; stage portable buckets until the receiver acknowledges them."""

    supports_lora = False
    wire_load_format = "safetensors_file"

    def _serialize_buckets(self, named_tensors):
        # Torch's CUDA IPC handles are not forward-compatible: the LMDeploy
        # environment may run an older Torch than Megatron. A local file avoids
        # serializing version-specific Tensor/Storage objects across that boundary.
        return serialize_tensor_bucket(named_tensors)
