"""Head layout of a delta-rule layer (GDN, KDA): counts and dims, globally or for one TP rank."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeltaRuleHeads:
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int

    @property
    def v_per_k(self) -> int:
        assert self.num_v_heads % self.num_k_heads == 0
        return self.num_v_heads // self.num_k_heads

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def qkv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def group_qkv_dim(self) -> int:
        return 2 * self.head_k_dim + self.v_per_k * self.head_v_dim

    @property
    def group_value_dim(self) -> int:
        return self.v_per_k * self.head_v_dim

    def local(self, tp_size: int) -> "DeltaRuleHeads":
        assert self.num_k_heads % tp_size == 0, f"{self.num_k_heads} key heads do not split across TP={tp_size}"
        return DeltaRuleHeads(
            self.num_k_heads // tp_size, self.num_v_heads // tp_size, self.head_k_dim, self.head_v_dim
        )
