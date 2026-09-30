from types import ModuleType
from typing import Any

import pytest
import torch


class TestRegisterCPUMemory:
    def test_each_buffer_is_registered_at_its_own_address_and_byte_size(
        self, p2p_transfer_utils: ModuleType, fake_transfer_engine: Any
    ) -> None:
        """A wrong address or length lets the RDMA write read outside the buffer it claims to send."""
        backing = torch.zeros(8)
        params = {"fp32_view": backing[2:5], "bf16": torch.zeros(5, dtype=torch.bfloat16)}

        registry = p2p_transfer_utils.register_cpu_memory(params, fake_transfer_engine)

        assert fake_transfer_engine.registered == [
            (backing.data_ptr() + 2 * 4, 3 * 4),
            (params["bf16"].data_ptr(), 5 * 2),
        ]
        assert registry == {
            "fp32_view": (backing.data_ptr() + 2 * 4, 3, 4),
            "bf16": (params["bf16"].data_ptr(), 5, 2),
        }

    def test_a_nonzero_registration_code_raises_and_names_the_weight(
        self, p2p_transfer_utils: ModuleType, fake_transfer_engine: Any
    ) -> None:
        """An unregistered buffer cannot be read by the transfer engine, so every later write would fail."""
        fake_transfer_engine.register_return_code = 5

        with pytest.raises(RuntimeError, match="register CPU memory failed for weight w, error: 5"):
            p2p_transfer_utils.register_cpu_memory({"w": torch.zeros(2)}, fake_transfer_engine)
